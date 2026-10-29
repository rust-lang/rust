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

### Sections

<!-- FIXME: Implementations -->
<!-- FIXME: Trait Implementations -->
<!-- FIXME: Implementors -->
<!-- FIXME: Auto Trait Implementations -->

#### Aliased Type

A type alias is expanded at compile time to its
[aliased type](https://doc.rust-lang.org/reference/items/type-aliases.html).
That may involve substituting some or all of the type parameters in the target
type with types provided by the type alias definition. The Aliased Type section
shows the result of this expansion, including the types of public fields or
variants, which may depend on those substitutions.

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
a dynamic interface appears at the top of the page composed of the [search]
interface, help screen, and [options].

[options]: read-documentation/in-doc-settings.html
[search]: read-documentation/search.md

Paths are supported as well, you can look for `Vec::new` or `Option::Some` or
even `module::module_child::another_child::struct::field`. Whitespace characters
are considered the same as `::`, so if you write `Vec    new`, it will be
considered the same as `Vec::new`.

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
