#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
htmldocck.py is a custom checker script for Rustdoc HTML outputs.

# How and why?

The principle is simple: This script receives a path to generated HTML
documentation and a "template" script, which has a series of check
commands like `@has` or `@matches`. Each command is used to check if
some pattern is present or not present in the particular file or in
a particular node of the HTML tree. In many cases, the template script
happens to be the source code given to rustdoc.

While it indeed is possible to test in smaller portions, it has been
hard to construct tests in this fashion and major rendering errors were
discovered much later. This script is designed to make black-box and
regression testing of Rustdoc easy. This does not preclude the needs for
unit testing, but can be used to complement related tests by quickly
showing the expected renderings.

In order to avoid one-off dependencies for this task, this script uses
a reasonably working HTML parser and the existing XPath implementation
from Python's standard library. Hopefully, we won't render
non-well-formed HTML.

# Commands

Commands start with an `@` followed by a command name (letters and
hyphens), and zero or more arguments separated by one or more whitespace
characters and optionally delimited with single or double quotes. The `@`
mark cannot be preceded by a non-whitespace character. Other lines
(including every text up to the first `@`) are ignored, but it is
recommended to avoid the use of `@` in the template file.

There are a number of supported commands:

* `@has PATH` checks for the existence of the given file.

  `PATH` is relative to the output directory. It can be given as `-`
  which repeats the most recently used `PATH`.

* `@hasraw PATH PATTERN` and `@matchesraw PATH PATTERN` checks
  for the occurrence of the given pattern `PATTERN` in the specified file.
  Only one occurrence of the pattern is enough.

  For `@hasraw`, `PATTERN` is a whitespace-normalized (every consecutive
  whitespace being replaced by one single space character) string.
  The entire file is also whitespace-normalized including newlines.

  For `@matchesraw`, `PATTERN` is a Python-supported regular expression.
  The file remains intact but the regexp is matched without the `MULTILINE`
  and `IGNORECASE` options. You can still use a prefix `(?m)` or `(?i)`
  to override them, and `\A` and `\Z` for definitely matching
  the beginning and end of the file.

  (The same distinction goes to other variants of these commands.)

* `@has PATH XPATH PATTERN` and `@matches PATH XPATH PATTERN` checks for
  the presence of the given XPath `XPATH` in the specified HTML file,
  and also the occurrence of the given pattern `PATTERN` in the matching
  node or attribute. Only one occurrence of the pattern in the match
  is enough.

  `PATH` should be a valid and well-formed HTML file. It does *not*
  accept arbitrary HTML5; it should have matching open and close tags
  and correct entity references at least.

  `XPATH` is an XPath expression to match. The XPath is fairly limited:
  `tag`, `*`, `.`, `//`, `..`, `[@attr]`, `[@attr='value']`, `[tag]`,
  `[POS]` (element located in given `POS`), `[last()-POS]`, `text()`
  and `@attr` (both as the last segment) are supported. Some examples:

  - `//pre` or `.//pre` matches any element with a name `pre`.
  - `//a[@href]` matches any element with an `href` attribute.
  - `//*[@class="impl"]//code` matches any element with a name `code`,
    which is an ancestor of some element which `class` attr is `impl`.
  - `//h1[@class="fqn"]/span[1]/a[last()]/@class` matches a value of
    `class` attribute in the last `a` element (can be followed by more
    elements that are not `a`) inside the first `span` in the `h1` with
    a class of `fqn`. Note that there cannot be any additional elements
    between them due to the use of `/` instead of `//`.

  Do not try to use non-absolute paths, it won't work due to the flawed
  ElementTree implementation. The script rejects them.

  For the text matches (i.e. paths not ending with `@attr`), any
  subelements are flattened into one string; this is handy for ignoring
  highlights for example. If you want to simply check for the presence of
  a given node or attribute, use an empty string (`""`) as a `PATTERN`.

* `@count PATH XPATH COUNT` checks for the occurrence of the given XPath
  in the specified file. The number of occurrences must match the given
  count.

* `@count PATH XPATH TEXT COUNT` checks for the occurrence of the given XPath
  with the given text in the specified file. The number of occurrences must
  match the given count.

* `@snapshot NAME PATH XPATH` creates a snapshot test named NAME.
  A snapshot test captures a subtree of the DOM, at the location
  determined by the XPath, and compares it to a pre-recorded value
  in a file. The file's name is the test's name with the `.rs` extension
  replaced with `.NAME.html`, where NAME is the snapshot's name.

  htmldocck supports the `--bless` option to accept the current subtree
  as expected, saving it to the file determined by the snapshot's name.
  compiletest's `--bless` flag is forwarded to htmldocck.

* `@has-dir PATH` checks for the existence of the given directory.

* `@files FOLDER_PATH [ENTRIES]`, checks that `FOLDER_PATH` contains exactly
  `[ENTRIES]`.

All conditions can be negated with `!`. `@!has foo/type.NoSuch.html`
checks if the given file does not exist, for example.

"""

from __future__ import absolute_import, print_function, unicode_literals

import codecs
import io
import sys
import os.path
import re
import shlex
from collections import namedtuple

try:
    from html.parser import HTMLParser
except ImportError:
    from HTMLParser import HTMLParser
try:
    from xml.etree import cElementTree as ET
except ImportError:
    from xml.etree import ElementTree as ET

try:
    from html.entities import name2codepoint
except ImportError:
    from htmlentitydefs import name2codepoint

# "void elements" (no closing tag) from the HTML Standard section 12.1.2
VOID_ELEMENTS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "keygen",
    "link",
    "menuitem",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}

# Python 2 -> 3 compatibility
try:
    unichr  # noqa: B018 FIXME: py2
except NameError:
    unichr = chr


channel = os.environ["DOC_RUST_LANG_ORG_CHANNEL"]

# Initialized in main
rust_test_path = None
bless = None


class CustomHTMLParser(HTMLParser):
    """simplified HTML parser.

    this is possible because we are dealing with very regular HTML from
    rustdoc; we only have to deal with i) void elements and ii) empty
    attributes."""

    def __init__(self, target=None):
        HTMLParser.__init__(self)
        self.__builder = target or ET.TreeBuilder()

    def handle_starttag(self, tag, attrs):
        attrs = {k: v or "" for k, v in attrs}
        self.__builder.start(tag, attrs)
        if tag in VOID_ELEMENTS:
            self.__builder.end(tag)

    def handle_endtag(self, tag):
        self.__builder.end(tag)

    def handle_startendtag(self, tag, attrs):
        attrs = {k: v or "" for k, v in attrs}
        self.__builder.start(tag, attrs)
        self.__builder.end(tag)

    def handle_data(self, data):
        self.__builder.data(data)

    def handle_entityref(self, name):
        self.__builder.data(unichr(name2codepoint[name]))

    def handle_charref(self, name):
        code = int(name[1:], 16) if name.startswith(("x", "X")) else int(name, 10)
        self.__builder.data(unichr(code))

    def close(self):
        HTMLParser.close(self)
        return self.__builder.close()


Command = namedtuple("Command", "negated cmd args lineno context")


class FailedCheck(Exception):
    pass


class InvalidCheck(Exception):
    pass


def concat_multi_lines(f):
    """returns a generator out of the file object, which
    - removes `\\` then `\n` then a shared prefix with the previous line then
      optional whitespace;
    - keeps a line number (starting from 0) of the first line being
      concatenated."""
    lastline = None  # set to the last line when the last line has a backslash
    firstlineno = None
    catenated = ""
    for lineno, line in enumerate(f):
        line = line.rstrip("\r\n")

        # strip the common prefix from the current line if needed
        if lastline is not None:
            common_prefix = os.path.commonprefix([line, lastline])
            line = line[len(common_prefix) :].lstrip()

        firstlineno = firstlineno or lineno
        if line.endswith("\\"):
            if lastline is None:
                lastline = line[:-1]
            catenated += line[:-1]
        else:
            yield firstlineno, catenated + line
            lastline = None
            firstlineno = None
            catenated = ""

    if lastline is not None:
        print_err(lineno, line, "Trailing backslash at the end of the file")


def get_known_directive_names():
    def filter_line(line):
        line = line.strip()
        return line.startswith('"') and (line.endswith('",') or line.endswith('"'))

    # Equivalent to `src/tools/compiletest/src/header.rs` constant of the same name.
    with open(
        os.path.join(
            # We go back to `src`.
            os.path.dirname(os.path.dirname(__file__)),
            "tools/compiletest/src/directive-list.rs",
        ),
        "r",
        encoding="utf8",
    ) as fd:
        content = fd.read()
        return [
            line.strip().replace('",', "").replace('"', "")
            for line in content.split("\n")
            if filter_line(line)
        ]


# To prevent duplicating the list of commmands between `compiletest` and `htmldocck`, we put
# it into a common file which is included in rust code and parsed here.
# FIXME: This setup is temporary until we figure out how to improve this situation.
#        See <https://github.com/rust-lang/rust/issues/125813#issuecomment-2141953780>.
KNOWN_DIRECTIVE_NAMES = get_known_directive_names()

LINE_PATTERN = re.compile(
    r"""
    //@\s+
    (?P<negated>!?)(?P<cmd>[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)
    (?P<args>.*)$
""",
    re.X | re.UNICODE,
)


def get_commands(template):
    with io.open(template, encoding="utf-8") as f:
        for lineno, line in concat_multi_lines(f):
            m = LINE_PATTERN.search(line)
            if not m:
                continue

            cmd = m.group("cmd")
            negated = m.group("negated") == "!"
            if not negated and cmd in KNOWN_DIRECTIVE_NAMES:
                continue
            args = m.group("args")
            if args and not args[:1].isspace():
                print_err(lineno, line, "Invalid template syntax")
                continue
            try:
                args = shlex.split(args)
            except UnicodeEncodeError:
                args = [
                    arg.decode("utf-8") for arg in shlex.split(args.encode("utf-8"))
                ]
            except Exception as exc:
                raise Exception("line {}: {}".format(lineno + 1, exc)) from None
            yield Command(
                negated=negated, cmd=cmd, args=args, lineno=lineno + 1, context=line
            )


def _flatten(node, acc):
    if node.text:
        acc.append(node.text)
    for e in node:
        _flatten(e, acc)
        if e.tail:
            acc.append(e.tail)


def flatten(node):
    acc = []
    _flatten(node, acc)
    return "".join(acc)


def make_xml(text):
    xml = ET.XML("<xml>%s</xml>" % text)
    return xml


def normalize_xpath(path):
    path = path.replace("{{channel}}", channel)
    if path.startswith("//"):
        return "." + path  # avoid warnings
    elif path.startswith(".//"):
        return path
    else:
        raise InvalidCheck(
            "Non-absolute XPath is not supported due to implementation issues"
        )


class CachedFiles(object):
    def __init__(self, root):
        self.root = root
        self.files = {}
        self.trees = {}
        self.last_path = None

    def resolve_path(self, path):
        if path != "-":
            path = os.path.normpath(path)
            self.last_path = path
            return path
        elif self.last_path is None:
            raise InvalidCheck("Tried to use the previous path in the first command")
        else:
            return self.last_path

    def get_absolute_path(self, path):
        return os.path.join(self.root, path)

    def get_file(self, path):
        path = self.resolve_path(path)
        if path in self.files:
            return self.files[path]

        abspath = self.get_absolute_path(path)
        if not (os.path.exists(abspath) and os.path.isfile(abspath)):
            raise FailedCheck("File does not exist {!r}".format(path))

        with io.open(abspath, encoding="utf-8") as f:
            data = f.read()
            self.files[path] = data
            return data

    def get_tree(self, path):
        path = self.resolve_path(path)
        if path in self.trees:
            return self.trees[path]

        abspath = self.get_absolute_path(path)
        if not (os.path.exists(abspath) and os.path.isfile(abspath)):
            raise FailedCheck("File does not exist {!r}".format(path))

        with io.open(abspath, encoding="utf-8") as f:
            try:
                tree = ET.fromstringlist(f.readlines(), CustomHTMLParser())
            except Exception as e:
                raise RuntimeError(  # noqa: B904 FIXME: py2
                    "Cannot parse an HTML file {!r}: {}".format(path, e)
                )
            self.trees[path] = tree
            return self.trees[path]

    def get_dir(self, path):
        path = self.resolve_path(path)
        abspath = self.get_absolute_path(path)
        if not (os.path.exists(abspath) and os.path.isdir(abspath)):
            raise FailedCheck("Directory does not exist {!r}".format(path))


def check_string(data, pat, regexp):
    pat = pat.replace("{{channel}}", channel)
    if not pat:
        return True  # special case a presence testing
    elif regexp:
        return re.search(pat, data, flags=re.UNICODE) is not None
    else:
        data = " ".join(data.split())
        pat = " ".join(pat.split())
        return pat in data


def check_tree_attr(tree, path, attr, pat, regexp):
    path = normalize_xpath(path)
    ret = False
    for e in tree.findall(path):
        if attr in e.attrib:
            value = e.attrib[attr]
        else:
            continue

        ret = check_string(value, pat, regexp)
        if ret:
            break
    return ret


# Returns the number of occurrences matching the regex (`regexp`) and the text (`pat`).
def check_tree_text(tree, path, pat, regexp, stop_at_first):
    path = normalize_xpath(path)
    match_count = 0
    try:
        for e in tree.findall(path):
            try:
                value = flatten(e)
            except KeyError:
                continue
            else:
                if check_string(value, pat, regexp):
                    match_count += 1
                    if stop_at_first:
                        break
    except Exception:
        print('Failed to get path "{}"'.format(path))
        raise
    return match_count


def get_tree_count(tree, path):
    path = normalize_xpath(path)
    return len(tree.findall(path))


def check_snapshot(snapshot_name, actual_tree, normalize_to_text):
    assert rust_test_path.endswith(".rs")
    snapshot_path = "{}.{}.{}".format(rust_test_path[:-3], snapshot_name, "html")
    try:
        with open(snapshot_path, "r") as snapshot_file:
            expected_str = snapshot_file.read().replace("{{channel}}", channel)
    except FileNotFoundError:
        if bless:
            expected_str = None
        else:
            raise FailedCheck("No saved snapshot value")  # noqa: B904 FIXME: py2

    if not normalize_to_text:
        actual_str = ET.tostring(actual_tree).decode("utf-8")
    else:
        actual_str = flatten(actual_tree)

    # Conditions:
    #  1. Is --bless
    #  2. Are actual and expected tree different
    #  3. Are actual and expected text different
    if (
        not expected_str
        or (
            not normalize_to_text
            and not compare_tree(make_xml(actual_str), make_xml(expected_str), stderr)
        )
        or (normalize_to_text and actual_str != expected_str)
    ):
        if bless:
            with open(snapshot_path, "w") as snapshot_file:
                actual_str = actual_str.replace(channel, "{{channel}}")
                snapshot_file.write(actual_str)
        else:
            print("--- expected ---\n")
            print(expected_str)
            print("\n\n--- actual ---\n")
            print(actual_str)
            print()
            raise FailedCheck("Actual snapshot value is different than expected")


# Adapted from https://github.com/formencode/formencode/blob/3a1ba9de2fdd494dd945510a4568a3afeddb0b2e/formencode/doctest_xml_compare.py#L72-L120
def compare_tree(x1, x2, reporter=None):
    if x1.tag != x2.tag:
        if reporter:
            reporter("Tags do not match: %s and %s" % (x1.tag, x2.tag))
        return False
    for name, value in x1.attrib.items():
        if x2.attrib.get(name) != value:
            if reporter:
                reporter(
                    "Attributes do not match: %s=%r, %s=%r"
                    % (name, value, name, x2.attrib.get(name))
                )
            return False
    for name in x2.attrib:
        if name not in x1.attrib:
            if reporter:
                reporter("x2 has an attribute x1 is missing: %s" % name)
            return False
    if not text_compare(x1.text, x2.text):
        if reporter:
            reporter("text: %r != %r" % (x1.text, x2.text))
        return False
    if not text_compare(x1.tail, x2.tail):
        if reporter:
            reporter("tail: %r != %r" % (x1.tail, x2.tail))
        return False
    cl1 = list(x1)
    cl2 = list(x2)
    if len(cl1) != len(cl2):
        if reporter:
            reporter("children length differs, %i != %i" % (len(cl1), len(cl2)))
        return False
    i = 0
    for c1, c2 in zip(cl1, cl2):
        i += 1
        if not compare_tree(c1, c2, reporter=reporter):
            if reporter:
                reporter("children %i do not match: %s" % (i, c1.tag))
            return False
    return True


def text_compare(t1, t2):
    if not t1 and not t2:
        return True
    if t1 == "*" or t2 == "*":
        return True
    return (t1 or "").strip() == (t2 or "").strip()


def stderr(*args):
    if sys.version_info.major < 3:
        file = codecs.getwriter("utf-8")(sys.stderr)
    else:
        file = sys.stderr

    print(*args, file=file)


def print_err(lineno, context, err, message=None):
    global ERR_COUNT
    ERR_COUNT += 1
    stderr("{}: {}".format(lineno, message or err))
    if message and err:
        stderr("\t{}".format(err))

    if context:
        stderr("\t{}".format(context))


def get_nb_matching_elements(cache, c, regexp, stop_at_first):
    tree = cache.get_tree(c.args[0])
    pat, sep, attr = c.args[1].partition("/@")
    if sep:  # attribute
        tree = cache.get_tree(c.args[0])
        return check_tree_attr(tree, pat, attr, c.args[2], False)
    else:  # normalized text
        pat = c.args[1]
        if pat.endswith("/text()"):
            pat = pat[:-7]
        return check_tree_text(
            cache.get_tree(c.args[0]), pat, c.args[2], regexp, stop_at_first
        )


def check_files_in_folder(c, cache, folder, files):
    files = files.strip()
    if not files.startswith("[") or not files.endswith("]"):
        raise InvalidCheck(
            "Expected list as second argument of {} (ie '[]')".format(c.cmd)
        )

    folder = cache.get_absolute_path(folder)

    # First we create a set of files to check if there are duplicates.
    files = shlex.split(files[1:-1].replace(",", ""))
    files_set = set()
    for file in files:
        if file in files_set:
            raise InvalidCheck("Duplicated file `{}` in {}".format(file, c.cmd))
        files_set.add(file)
    folder_set = set([f for f in os.listdir(folder) if f != "." and f != ".."])

    # Then we remove entries from both sets (we clone `folder_set` so we can iterate it while
    # removing its elements).
    for entry in set(folder_set):
        if entry in files_set:
            files_set.remove(entry)
            folder_set.remove(entry)

    error = 0
    if len(files_set) != 0:
        print_err(
            c.lineno,
            c.context,
            "Entries not found in folder `{}`: `{}`".format(folder, files_set),
        )
        error += 1
    if len(folder_set) != 0:
        print_err(
            c.lineno,
            c.context,
            "Extra entries in folder `{}`: `{}`".format(folder, folder_set),
        )
        error += 1
    return error == 0


ERR_COUNT = 0


def check_command(c, cache):
    try:
        cerr = ""
        if c.cmd in ["has", "hasraw", "matches", "matchesraw"]:  # string test
            regexp = c.cmd.startswith("matches")

            # has <path> = file existence
            if len(c.args) == 1 and not regexp and "raw" not in c.cmd:
                try:
                    cache.get_file(c.args[0])
                    ret = True
                except FailedCheck as err:
                    cerr = str(err)
                    ret = False
            # hasraw/matchesraw <path> <pat> = string test
            elif len(c.args) == 2 and "raw" in c.cmd:
                cerr = "`PATTERN` did not match"
                ret = check_string(cache.get_file(c.args[0]), c.args[1], regexp)
            # has/matches <path> <pat> <match> = XML tree test
            elif len(c.args) == 3 and "raw" not in c.cmd:
                cerr = "`XPATH PATTERN` did not match"
                ret = get_nb_matching_elements(cache, c, regexp, True) != 0
            else:
                raise InvalidCheck("Invalid number of {} arguments".format(c.cmd))

        elif c.cmd == "files":  # check files in given folder
            if len(c.args) != 2:  # files <folder path> <file list>
                raise InvalidCheck("Invalid number of {} arguments".format(c.cmd))
            elif c.negated:
                raise InvalidCheck("{} doesn't support negative check".format(c.cmd))
            ret = check_files_in_folder(c, cache, c.args[0], c.args[1])

        elif c.cmd == "count":  # count test
            if len(c.args) == 3:  # count <path> <pat> <count> = count test
                expected = int(c.args[2])
                found = get_tree_count(cache.get_tree(c.args[0]), c.args[1])
                cerr = "Expected {} occurrences but found {}".format(expected, found)
                ret = expected == found
            elif len(c.args) == 4:  # count <path> <pat> <text> <count> = count test
                expected = int(c.args[3])
                found = get_nb_matching_elements(cache, c, False, False)
                cerr = "Expected {} occurrences but found {}".format(expected, found)
                ret = found == expected
            else:
                raise InvalidCheck("Invalid number of {} arguments".format(c.cmd))

        elif c.cmd == "snapshot":  # snapshot test
            if len(c.args) == 3:  # snapshot <snapshot-name> <html-path> <xpath>
                [snapshot_name, html_path, pattern] = c.args
                tree = cache.get_tree(html_path)
                xpath = normalize_xpath(pattern)
                normalize_to_text = False
                if xpath.endswith("/text()"):
                    xpath = xpath[:-7]
                    normalize_to_text = True

                subtrees = tree.findall(xpath)
                if len(subtrees) == 1:
                    [subtree] = subtrees
                    try:
                        check_snapshot(snapshot_name, subtree, normalize_to_text)
                        ret = True
                    except FailedCheck as err:
                        cerr = str(err)
                        ret = False
                elif len(subtrees) == 0:
                    raise FailedCheck("XPATH did not match")
                else:
                    raise FailedCheck(
                        "Expected 1 match, but found {}".format(len(subtrees))
                    )
            else:
                raise InvalidCheck("Invalid number of {} arguments".format(c.cmd))

        elif c.cmd == "has-dir":  # has-dir test
            if len(c.args) == 1:  # has-dir <path> = has-dir test
                try:
                    cache.get_dir(c.args[0])
                    ret = True
                except FailedCheck as err:
                    cerr = str(err)
                    ret = False
            else:
                raise InvalidCheck("Invalid number of {} arguments".format(c.cmd))

        elif c.cmd == "valid-html":
            raise InvalidCheck("Unimplemented valid-html")

        elif c.cmd == "valid-links":
            raise InvalidCheck("Unimplemented valid-links")

        else:
            raise InvalidCheck("Unrecognized {}".format(c.cmd))

        if ret == c.negated:
            raise FailedCheck(cerr)

    except FailedCheck as err:
        message = "{}{} check failed".format("!" if c.negated else "", c.cmd)
        print_err(c.lineno, c.context, str(err), message)
    except InvalidCheck as err:
        print_err(c.lineno, c.context, str(err))


def check(target, commands):
    cache = CachedFiles(target)
    for c in commands:
        check_command(c, cache)


if __name__ == "__main__":
    if len(sys.argv) not in [3, 4]:
        stderr("Usage: {} <doc dir> <template> [--bless]".format(sys.argv[0]))
        raise SystemExit(1)

    rust_test_path = sys.argv[2]
    if len(sys.argv) > 3 and sys.argv[3] == "--bless":
        bless = True
    else:
        # We only support `--bless` at the end of the arguments.
        # This assert is to prevent silent failures.
        assert "--bless" not in sys.argv
        bless = False
    check(sys.argv[1], get_commands(rust_test_path))
    if ERR_COUNT:
        stderr("\nEncountered {} errors".format(ERR_COUNT))
        raise SystemExit(1)
