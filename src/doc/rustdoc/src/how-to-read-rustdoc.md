# How to read rustdoc output

Rustdoc's HTML output includes a friendly and useful navigation interface which
makes it easier for users to navigate and understand your code.
This chapter covers the major features of that interface,
and is a great starting point for documentation authors and users alike.

## Structure

The `rustdoc` output is divided into three sections.
Along the left side of each page is a quick navigation bar,
which shows contextual information about the current entry.
The rest of the page is taken up by the search interface at the top
and the documentation for the current item below that.

## The Item Documentation

The majority of the screen is taken up with the documentation text for the item
currently being viewed.
At the top is some at-a-glance info and controls:

- the type and name of the item,
  such as "Struct `std::time::Duration`",
- a button to copy the item's path to the clipboard,
  which is a clipboard item
- a button to collapse or expand the top-level documentation for that item
  (`[+]` or `[-]`),
- a link to the source code (`[src]`),
  if [configured](write-documentation/the-doc-attribute.html#html_no_source),
  and present (the source may not be available if
  the documentation was created with `cargo doc --no-deps`),
- and the version in which the item became stable,
  if it's a stable item in the standard library.

Below this is the main documentation for the item,
including a definition or function signature if appropriate,
followed by a list of fields or variants for Rust types.
Finally, the page lists associated functions and trait implementations,
including automatic and blanket implementations that `rustdoc` knows about.

### Navigation

Subheadings, variants, fields, and many other things in this documentation
are anchors and can be clicked on and deep-linked to,
which is a great way to communicate exactly what you're talking about.
The typographical character "§" appears next to lines with anchors on them
when hovered or given keyboard focus.

## The Navigation Bar

For example, when looking at documentation for the crate root,
it shows all the crates documented in the documentation bundle,
and quick links to the modules, structs, traits, functions, and macros available
from the current crate.
At the top, it displays a [configurable logo](write-documentation/the-doc-attribute.html#html_logo_url)
alongside the current crate's name and version,
or the current item whose documentation is being displayed.

## The Theme Picker and Search Interface

When viewing `rustdoc`'s output in a browser with JavaScript enabled,
a dynamic interface appears at the top of the page composed of the search
interface, help screen, and options.

### The Search Interface

Typing in the search bar instantly searches the available documentation for
the string entered with a fuzzy matching algorithm that is tolerant of minor
typos.

By default, the search results given are "In Names",
meaning that the fuzzy match is made against the names of items.
Matching names are shown on the left, and the first few words of their
descriptions are given on the right.
By clicking an item, you will navigate to its particular documentation.

There are two other sets of results, shown as tabs in the search results pane.
"In Parameters" shows matches for the string in the types of parameters to
functions, and "In Return Types" shows matches in the return types of functions.
Both are very useful when looking for a function whose name you can't quite
bring to mind when you know the type you have or want.

Names in the search interface can be prefixed with an item type followed by a
colon (such as `mod:`) to restrict the results to just that kind of item. Also,
searching for `println!` will search for a macro named `println`, just like
searching for `macro:println` does.

Function signature searches can query generics, wrapped in angle brackets, and
traits are normalized like types in the search engine. For example, a function
with the signature `fn my_function<I: Iterator<Item=u32>>(input: I) -> usize`
can be matched with the following queries:

* `Iterator<u32> -> usize`
* `trait:Iterator<primitive:u32> -> primitive:usize`
* `Iterator -> usize`

Generics and function parameters are order-agnostic, but sensitive to nesting
and number of matches. For example, a function with the signature
`fn read_all(&mut self: impl Read) -> Result<Vec<u8>, Error>`
will match these queries:

* `Read -> Result<Vec<u8>, Error>`
* `Read -> Result<Error, Vec>`
* `Read -> Result<Vec<u8>>`

But it *does not* match `Result<Vec, u8>` or `Result<u8<Vec>>`.

Function signature searches also support arrays and slices. The explicit name
`primitive:slice<u8>` and `primitive:array<u8>` can be used to match a slice
or array of bytes, while square brackets `[u8]` will match either one. Empty
square brackets, `[]`, will match any slice regardless of what it contains.

### Shortcuts

Pressing `S` while focused elsewhere on the page will move focus to the
search bar, and pressing `?` shows the help screen,
which includes all these shortcuts and more.

When the search results are focused,
the left and right arrows move between tabs and the up and down arrows move
among the results.
Pressing the enter or return key opens the highlighted result.

When looking at the documentation for an item, the plus and minus keys expand
and collapse all sections in the document.
