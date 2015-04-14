- Feature Name: rename_soft_link_to_symlink
- Start Date: 2015-04-09
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Rename `std::fs::soft_link` into platform-specific versions:
`std::os::unix::fs::symlink`, `std::os::windows::fs::symlink_file`, and
`std::os::windows::fs::symlink_dir`.

# Motivation

Windows Vista introduced the ability to create symbolic links, in order to
[provide compatibility with applications ported from Unix](https://msdn.microsoft.com/en-us/library/windows/desktop/aa365680%28v=vs.85%29.aspx):

> Symbolic links are designed to aid in migration and application
> compatibility with UNIX operating systems. Microsoft has implemented its
> symbolic links to function just like UNIX links.

However, symbolic links on Windows behave differently enough than symbolic
links on Unix family operating systems that you can't, in general, assume that
code that works on one will work on the other.  On Unix family operating
systems, a symbolic link may refer to either a directory or a file, and which
one is determined when it is resolved to an actual file.  On Windows, you must
specify at the time of creation whether a symbolic link refers to a file or
directory.

In addition, an arbitrary process on Windows is not allowed to create a
symlink; you need to have [particular privileges][1] in order to be able to do
so; while on Unix, ordinary users can create symlinks, and any additional
security policy (such as [Grsecurity][2]) generally restricts
whether applications follow symlinks, not whether a user can create them.

[1]: (https://technet.microsoft.com/en-us/library/cc766301%28WS.10%29.aspx) in order to be able to do
[2]: https://en.wikibooks.org/wiki/Grsecurity/Appendix/Grsecurity_and_PaX_Configuration_Options#Linking_restrictions

Thus, there needs to be a way to distinguish between the two operations on
Windows, but that distinction is meaningless on Unix, and any code that deals
with symlinks on Windows will need to depend on having appropriate privilege
or have some way of obtaining appropriate privilege, which is all quite
platform specific.

These two facts mean that it is unlikely that arbitrary code dealing with
symbolic links will be portable between Windows and Unix.  Rather than trying
to support both under one API, it would be better to provide platform specific
APIs, making it much more clear upon inspection where portability issues may
arise.

In addition, the current name `soft_link` is fairly non-standard.  At some
point in the split up version of rust-lang/rfcs#517, `std::fs::symlink` was
renamed to `sym_link` and then to `soft_link`.

The new name is somewhat surprising and can be difficult to find.  After a
poll of a number of different platforms and languages, every one appears to
contain `symlink`, `symbolic_link`, or some camel case variant of those for
their equivalent API.  Every piece of formal documentation found, for
both Windows and various Unix like platforms, used "symbolic link" exclusively
in prose.

Here are the names I found for this functionality on various platforms,
libraries, and languages:

* [POSIX/Single Unix Specification](http://pubs.opengroup.org/onlinepubs/009695399/functions/symlink.html): `symlink`
* [Windows](https://msdn.microsoft.com/en-us/library/windows/desktop/aa365680%28v=vs.85%29.aspx): `CreateSymbolicLink`
* [Objective-C/Swift](https://developer.apple.com/library/ios/documentation/Cocoa/Reference/Foundation/Classes/NSFileManager_Class/index.html#//apple_ref/occ/instm/NSFileManager/createSymbolicLinkAtPath:withDestinationPath:error:): `createSymbolicLinkAtPath:withDestinationPath:error:`
* [Java](https://docs.oracle.com/javase/7/docs/api/java/nio/file/Files.html): `createSymbolicLink`
* [C++ (Boost/draft standard)](http://en.cppreference.com/w/cpp/experimental/fs): `create_symlink`
* [Ruby](http://ruby-doc.org/core-2.2.0/File.html): `symlink`
* [Python](https://docs.python.org/2/library/os.html#os.symlink): `symlink`
* [Perl](http://perldoc.perl.org/functions/symlink.html): `symlink`
* [PHP](https://php.net/manual/en/function.symlink.php): `symlink`
* [Delphi](http://docwiki.embarcadero.com/Libraries/XE7/en/System.SysUtils.FileCreateSymLink): `FileCreateSymLink`
* PowerShell has no official version, but several community cmdlets ([one example](http://stackoverflow.com/questions/894430/powershell-hard-and-soft-links/894651#894651), [another example](https://gallery.technet.microsoft.com/scriptcenter/New-SymLink-60d2531e)) are named `New-SymLink`

The term "soft link", probably as a contrast with "hard link", is found
frequently in informal descriptions, but almost always in the form of a
parenthetical of an alternate phrase, such as "a symbolic link (or soft
link)".  I could not find it used in any formal documentation or APIs outside
of Rust.

The name `soft_link` was chosen to be shorter than `symbolic_link`, but
without using Unix specific jargon like `symlink`, to not give undue weight to
one platform over the other.  However, based on the evidence above it doesn't
have any precedent as a formal name for the concept or API.

Furthermore, even on Windows, the name for the [reparse point tag used][3] to
represent symbolic links is `IO_REPARSE_TAG_SYMLINK`.

[3]: https://msdn.microsoft.com/en-us/library/windows/desktop/aa365511%28v=vs.85%29.aspx

If you do a Google search for "[windows symbolic link](https://www.google.com/search?q=windows+symbolic+link&ie=utf-8&oe=utf-8)" or "[windows soft link](https://www.google.com/search?q=windows+soft+link&ie=utf-8&oe=utf-8)",
many of the documents you find start using "symlink" after introducing the
concept, so it seems to be a fairly common abbreviation for the full name even
among Windows developers and users.

# Detailed design

Move `std::fs::soft_link` to `std::os::unix::fs::symlink`, and create
`std::os::windows::fs::symlink_file` and `std::os::windows::fs::symlink_dir`
that call `CreateSymbolicLink` with the appropriate arguments.

Keep a deprecated compatibility wrapper `std::fs::soft_link` which wraps
`std::os::unix::fs::symlink` or `std::os::windows::fs::symlink_file`,
depending on the platform (as that is the current behavior of
`std::fs::softlink`, to create a file symbolic link).

# Drawbacks

This deprecates a stable API during the 1.0.0 beta, leaving an extra wrapper
around.

# Alternatives

* Have a cross platform `symlink` and `symlink_dir`, that do the same thing on
  Unix but differ on Windows.  This has the drawback of invisible
  compatibility hazards; code that works on Unix using `symlink` may fail
  silently on Windows, as creating the wrong type of symlink may succeed but
  it may not be interpreted properly once a destination file of the other type
  is created.
* Have a cross platform `symlink` that detects the type of the destination
  on Windows.  This is not always possible as it's valid to create dangling
  symbolic links.
* Have `symlink`, `symlink_dir`, and `symlink_file` all cross-platform, where
  the first dispatches based on the destination file type, and the latter two
  panic if called with the wrong destination file type.  Again, this is not
  always possible as it's valid to create dangling symbolic links.
* Rather than having two separate functions on Windows, you could have a
  separate parameter on Windows to specify the type of link to create;
  `symlink("a", "b", FILE_SYMLINK)` vs `symlink("a", "b", DIR_SYMLINK)`.
  However, having a `symlink` that had different arity on Unix and Windows
  would likely be confusing, and since there are only the two possible
  choices, simply having two functions seems like a much simpler solution.

Other choices for the naming convention would be:

* The status quo, `soft_link`
* The original proposal from rust-lang/rfcs#517, `sym_link`
* The full name, `symbolic_link`

The first choice is non-obvious, for people coming from either Windows or
Unix.  It is a classic compromise, that makes everyone unhappy.

`sym_link` is slightly more consistent with the complementary `hard_link`
function, and treating "sym link" as two separate words has some precedent in
two of the Windows-targetted APIs, Delphi and some of the PowerShell cmdlets
observed.  However, I have not found any other snake case API that uses that,
and only a couple of Windows-specific APIs that use it in camel case; most
usage prefers the single word "symlink" to the two word "sym link" as the
abbreviation.

The full name `symbolic_link`, is a bit long and cumbersome compared to most
of the rest of the API, but is explicit and is the term used in prose to
describe the concept everywhere, so shouldn't emphasize any one platform over
the other.  However, unlike all other operations for creating a file or
directory (`open`, `create`, `create_dir`, etc), it is a noun, not a verb.
When used as a verb, it would be called "symbolically link", but that sounds
quite odd in the context of an API: `symbolically_link("a", "b")`.  "symlink",
on the other hand, can act as either a noun or a verb.

It would be possible to prefix any of the forms above that read as a noun with
`create_`, such as `create_symlink`, `create_sym_link`,
`create_symbolic_link`.  This adds further to the verbosity, though it is
consisted with `create_dir`; you would probably need to also rename
`hard_link` to `create_hard_link` for consistency, and this seems like a lot
of churn and extra verbosity for not much benefit, as `symlink` and
`hard_link` already act as verbs on their own.  If you picked this, then the
Windows versions would need to be named `create_file_symlink` and
`create_dir_symlink` (or the variations with `sym_link` or `symbolic_link`).

# Unresolved questions

If we deprecate `soft_link` now, early in the beta cycle, would it be
acceptable to remove it rather than deprecate it before 1.0.0, thus avoiding a
permanently stable but deprecated API right out the gate?
