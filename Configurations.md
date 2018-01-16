# Configuring Rustfmt

Rustfmt is designed to be very configurable. You can create a TOML file called `rustfmt.toml` or `.rustfmt.toml`, place it in the project or any other parent directory and it will apply the options in that file.

A possible content of `rustfmt.toml` or `.rustfmt.toml` might look like this:

```toml
indent_style = "Block"
reorder_imported_names = true
```

Each configuration option is either stable or unstable.
Stable options can be used directly, while unstable options are opt-in.
To enable unstable options, set `unstable_features = true` in `rustfmt.toml` or pass `--unstable-options` to rustfmt.

# Configuration Options

Below you find a detailed visual guide on all the supported configuration options of rustfmt:


## `indent_style`

Indent on expressions or items.

- **Default value**: `"Block"`
- **Possible values**: `"Block"`, `"Visual"`
- **Stable**: No

### Array

#### `"Block"` (default):

```rust
let lorem = vec![
    "ipsum",
    "dolor",
    "sit",
    "amet",
    "consectetur",
    "adipiscing",
    "elit",
];
```

#### `"Visual"`:

```rust
let lorem = vec!["ipsum",
                 "dolor",
                 "sit",
                 "amet",
                 "consectetur",
                 "adipiscing",
                 "elit"];
```

### Control flow

#### `"Block"` (default):

```rust
if lorem_ipsum &&
    dolor_sit &&
    amet_consectetur
{
    // ...
}
```

#### `"Visual"`:

```rust
if lorem_ipsum &&
   dolor_sit &&
   amet_consectetur {
    // ...
}
```

See also: [`control_brace_style`](#control_brace_style).

### Function arguments

#### `"Block"` (default):

```rust
fn lorem() {}

fn lorem(ipsum: usize) {}

fn lorem(
    ipsum: usize,
    dolor: usize,
    sit: usize,
    amet: usize,
    consectetur: usize,
    adipiscing: usize,
    elit: usize,
) {
    // body
}
```

#### `"Visual"`:

```rust
fn lorem() {}

fn lorem(ipsum: usize) {}

fn lorem(ipsum: usize,
         dolor: usize,
         sit: usize,
         amet: usize,
         consectetur: usize,
         adipiscing: usize,
         elit: usize) {
    // body
}
```

### Function calls

#### `"Block"` (default):

```rust
lorem(
    "lorem",
    "ipsum",
    "dolor",
    "sit",
    "amet",
    "consectetur",
    "adipiscing",
    "elit",
);
```

#### `"Visual"`:

```rust
lorem("lorem",
      "ipsum",
      "dolor",
      "sit",
      "amet",
      "consectetur",
      "adipiscing",
      "elit");
```

### Generics

#### `"Block"` (default):

```rust
fn lorem<
    Ipsum: Eq = usize,
    Dolor: Eq = usize,
    Sit: Eq = usize,
    Amet: Eq = usize,
    Adipiscing: Eq = usize,
    Consectetur: Eq = usize,
    Elit: Eq = usize
>(
    ipsum: Ipsum,
    dolor: Dolor,
    sit: Sit,
    amet: Amet,
    adipiscing: Adipiscing,
    consectetur: Consectetur,
    elit: Elit,
) -> T {
    // body
}
```

#### `"Visual"`:

```rust
fn lorem<Ipsum: Eq = usize,
         Dolor: Eq = usize,
         Sit: Eq = usize,
         Amet: Eq = usize,
         Adipiscing: Eq = usize,
         Consectetur: Eq = usize,
         Elit: Eq = usize>
    (ipsum: Ipsum,
     dolor: Dolor,
     sit: Sit,
     amet: Amet,
     adipiscing: Adipiscing,
     consectetur: Consectetur,
     elit: Elit)
     -> T {
    // body
}
```

#### Struct

#### `"Block"` (default):

```rust
let lorem = Lorem {
    ipsum: dolor,
    sit: amet,
};
```

#### `"Visual"`:

```rust
let lorem = Lorem { ipsum: dolor,
                    sit: amet, };
```

See also: [`struct_lit_single_line`](#struct_lit_single_line), [`indent_style`](#indent_style).

### Where predicates

#### `"Block"` (default):

```rust
fn lorem<Ipsum, Dolor, Sit, Amet>() -> T
where
    Ipsum: Eq,
    Dolor: Eq,
    Sit: Eq,
    Amet: Eq
{
    // body
}
```

#### `"Visual"`:

```rust
fn lorem<Ipsum, Dolor, Sit, Amet>() -> T
    where Ipsum: Eq,
          Dolor: Eq,
          Sit: Eq,
          Amet: Eq
{
    // body
}
```

## `use_small_heuristics`

Whether to use different formatting for items and expressions if they satisfy a heuristic notion of 'small'.

- **Default value**: `true`
- **Possible values**: `true`, `false`
- **Stable**: No

#### `true` (default):

```rust
enum Lorem {
    Ipsum,
    Dolor(bool),
    Sit { amet: Consectetur, adipiscing: Elit },
}

fn main() {
    lorem(
        "lorem",
        "ipsum",
        "dolor",
        "sit",
        "amet",
        "consectetur",
        "adipiscing",
    );

    let lorem = Lorem { ipsum: dolor, sit: amet };

    let lorem = if ipsum { dolor } else { sit };
}
```

#### `false`:

```rust
enum Lorem {
    Ipsum,
    Dolor(bool),
    Sit {
        amet: Consectetur,
        adipiscing: Elit,
    },
}

fn main() {
    lorem("lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing");

    let lorem = Lorem {
        ipsum: dolor,
        sit: amet,
    };

    let lorem = if ipsum {
        dolor
    } else {
        sit
    };
}
```

## `binop_separator`

Where to put a binary operator when a binary expression goes multiline.

- **Default value**: `"Front"`
- **Possible values**: `"Front"`, `"Back"`
- **Stable**: No

#### `"Front"` (default):

```rust
let or = foo
    || bar
    || foobar;

let sum = 1234
    + 5678
    + 910;

let range = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    ..bbbbbbbbbbbbbbbbbbbbbbbbbbbbb;
```

#### `"Back"`:

```rust
fn main() {
    let or = foofoofoofoofoofoofoofoofoofoofoofoofoofoofoofoo ||
        barbarbarbarbarbarbarbarbarbarbarbarbarbarbarbar;

    let sum = 123456789012345678901234567890 + 123456789012345678901234567890 +
        123456789012345678901234567890;

    let range = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa..
        bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb;
}
```

## `combine_control_expr`

Combine control expressions with function calls.

- **Default value**: `true`
- **Possible values**: `true`, `false`
- **Stable**: No

#### `true` (default):

```rust
fn example() {
    // If
    foo!(if x {
        foo();
    } else {
        bar();
    });

    // IfLet
    foo!(if let Some(..) = x {
        foo();
    } else {
        bar();
    });

    // While
    foo!(while x {
        foo();
        bar();
    });

    // WhileLet
    foo!(while let Some(..) = x {
        foo();
        bar();
    });

    // ForLoop
    foo!(for x in y {
        foo();
        bar();
    });

    // Loop
    foo!(loop {
        foo();
        bar();
    });
}
```

#### `false`:

```rust
fn example() {
    // If
    foo!(
        if x {
            foo();
        } else {
            bar();
        }
    );

    // IfLet
    foo!(
        if let Some(..) = x {
            foo();
        } else {
            bar();
        }
    );

    // While
    foo!(
        while x {
            foo();
            bar();
        }
    );

    // WhileLet
    foo!(
        while let Some(..) = x {
            foo();
            bar();
        }
    );

    // ForLoop
    foo!(
        for x in y {
            foo();
            bar();
        }
    );

    // Loop
    foo!(
        loop {
            foo();
            bar();
        }
    );
}
```

## `comment_width`

Maximum length of comments. No effect unless`wrap_comments = true`.

- **Default value**: `80`
- **Possible values**: any positive integer
- **Stable**: No

**Note:** A value of `0` results in [`wrap_comments`](#wrap_comments) being applied regardless of a line's width.

#### `80` (default; comments shorter than `comment_width`):
```rust
// Lorem ipsum dolor sit amet, consectetur adipiscing elit.
```

#### `60` (comments longer than `comment_width`):
```rust
// Lorem ipsum dolor sit amet,
// consectetur adipiscing elit.
```

See also [`wrap_comments`](#wrap_comments).

## `condense_wildcard_suffixes`

Replace strings of _ wildcards by a single .. in tuple patterns

- **Default value**: `false`
- **Possible values**: `true`, `false`
- **Stable**: No

#### `false` (default):

```rust
let (lorem, ipsum, _, _) = (1, 2, 3, 4);
```

#### `true`:

```rust
let (lorem, ipsum, ..) = (1, 2, 3, 4);
```

## `control_brace_style`

Brace style for control flow constructs

- **Default value**: `"AlwaysSameLine"`
- **Possible values**: `"AlwaysNextLine"`, `"AlwaysSameLine"`, `"ClosingNextLine"`
- **Stable**: No

#### `"AlwaysSameLine"` (default):

```rust
if lorem {
    println!("ipsum!");
} else {
    println!("dolor!");
}
```

#### `"AlwaysNextLine"`:

```rust
if lorem
{
    println!("ipsum!");
}
else
{
    println!("dolor!");
}
```

#### `"ClosingNextLine"`:

```rust
if lorem {
    println!("ipsum!");
}
else {
    println!("dolor!");
}
```

## `disable_all_formatting`

Don't reformat anything

- **Default value**: `false`
- **Possible values**: `true`, `false`
- **Stable**: No

## `error_on_line_overflow`

Error if unable to get all lines within `max_width`

- **Default value**: `true`
- **Possible values**: `true`, `false`
- **Stable**: No

See also [`max_width`](#max_width).

## `error_on_line_overflow_comments`

Error if unable to get all comment lines within `comment_width`.

- **Default value**: `true`
- **Possible values**: `true`, `false`
- **Stable**: No

See also [`comment_width`](#comment_width).

## `fn_args_density`

Argument density in functions

- **Default value**: `"Tall"`
- **Possible values**: `"Compressed"`, `"Tall"`, `"Vertical"`
- **Stable**: No

#### `"Tall"` (default):

```rust
trait Lorem {
    fn lorem(ipsum: Ipsum, dolor: Dolor, sit: Sit, amet: Amet);

    fn lorem(ipsum: Ipsum, dolor: Dolor, sit: Sit, amet: Amet) {
        // body
    }

    fn lorem(
        ipsum: Ipsum,
        dolor: Dolor,
        sit: Sit,
        amet: Amet,
        consectetur: Consectetur,
        adipiscing: Adipiscing,
        elit: Elit,
    );

    fn lorem(
        ipsum: Ipsum,
        dolor: Dolor,
        sit: Sit,
        amet: Amet,
        consectetur: Consectetur,
        adipiscing: Adipiscing,
        elit: Elit,
    ) {
        // body
    }
}
```

#### `"Compressed"`:

```rust
trait Lorem {
    fn lorem(ipsum: Ipsum, dolor: Dolor, sit: Sit, amet: Amet);

    fn lorem(ipsum: Ipsum, dolor: Dolor, sit: Sit, amet: Amet) {
        // body
    }

    fn lorem(
        ipsum: Ipsum, dolor: Dolor, sit: Sit, amet: Amet, consectetur: Consectetur,
        adipiscing: Adipiscing, elit: Elit,
    );

    fn lorem(
        ipsum: Ipsum, dolor: Dolor, sit: Sit, amet: Amet, consectetur: Consectetur,
        adipiscing: Adipiscing, elit: Elit,
    ) {
        // body
    }
}
```

#### `"Vertical"`:

```rust
trait Lorem {
    fn lorem(ipsum: Ipsum,
             dolor: Dolor,
             sit: Sit,
             amet: Amet);

    fn lorem(ipsum: Ipsum,
             dolor: Dolor,
             sit: Sit,
             amet: Amet) {
        // body
    }

    fn lorem(ipsum: Ipsum,
             dolor: Dolor,
             sit: Sit,
             amet: Amet,
             consectetur: Consectetur,
             adipiscing: Adipiscing,
             elit: Elit);

    fn lorem(ipsum: Ipsum,
             dolor: Dolor,
             sit: Sit,
             amet: Amet,
             consectetur: Consectetur,
             adipiscing: Adipiscing,
             elit: Elit) {
        // body
    }
}
```


## `brace_style`

Brace style for items

- **Default value**: `"SameLineWhere"`
- **Possible values**: `"AlwaysNextLine"`, `"PreferSameLine"`, `"SameLineWhere"`
- **Stable**: No

### Functions

#### `"SameLineWhere"` (default):

```rust
fn lorem() {
    // body
}

fn lorem(ipsum: usize) {
    // body
}

fn lorem<T>(ipsum: T)
where
    T: Add + Sub + Mul + Div,
{
    // body
}
```

#### `"AlwaysNextLine"`:

```rust
fn lorem()
{
    // body
}

fn lorem(ipsum: usize)
{
    // body
}

fn lorem<T>(ipsum: T)
where
    T: Add + Sub + Mul + Div,
{
    // body
}
```

#### `"PreferSameLine"`:

```rust
fn lorem() {
    // body
}

fn lorem(ipsum: usize) {
    // body
}

fn lorem<T>(ipsum: T)
where
    T: Add + Sub + Mul + Div, {
    // body
}
```

### Structs and enums

#### `"SameLineWhere"` (default):

```rust
struct Lorem {
    ipsum: bool,
}

struct Dolor<T>
    where T: Eq
{
    sit: T,
}
```

#### `"AlwaysNextLine"`:

```rust
struct Lorem
{
    ipsum: bool,
}

struct Dolor<T>
    where T: Eq
{
    sit: T,
}
```

#### `"PreferSameLine"`:

```rust
struct Lorem {
    ipsum: bool,
}

struct Dolor<T>
    where T: Eq {
    sit: T,
}
```


## `empty_item_single_line`

Put empty-body functions and impls on a single line

- **Default value**: `true`
- **Possible values**: `true`, `false`
- **Stable**: No

#### `true` (default):

```rust
fn lorem() {}

impl Lorem {}
```

#### `false`:

```rust
fn lorem() {
}

impl Lorem {
}
```

See also [`brace_style`](#brace_style), [`control_brace_style`](#control_brace_style).


## `fn_single_line`

Put single-expression functions on a single line

- **Default value**: `false`
- **Possible values**: `true`, `false`
- **Stable**: No

#### `false` (default):

```rust
fn lorem() -> usize {
    42
}

fn lorem() -> usize {
    let ipsum = 42;
    ipsum
}
```

#### `true`:

```rust
fn lorem() -> usize { 42 }

fn lorem() -> usize {
    let ipsum = 42;
    ipsum
}
```

See also [`control_brace_style`](#control_brace_style).


## `where_single_line`

To force single line where layout

- **Default value**: `false`
- **Possible values**: `true`, `false`
- **Stable**: No

#### `false` (default):

```rust
impl<T> Lorem for T
where
    Option<T>: Ipsum,
{
    ...
}
```

#### `true`:

```rust
impl<T> Lorem for T
where Option<T>: Ipsum {
    ...
}
```

See also [`brace_style`](#brace_style), [`control_brace_style`](#control_brace_style).


## `force_explicit_abi`

Always print the abi for extern items

- **Default value**: `true`
- **Possible values**: `true`, `false`
- **Stable**: Yes

**Note:** Non-"C" ABIs are always printed. If `false` then "C" is removed.

#### `true` (default):

```rust
extern "C" {
    pub static lorem: c_int;
}
```

#### `false`:

```rust
extern {
    pub static lorem: c_int;
}
```

## `format_strings`

Format string literals where necessary

- **Default value**: `false`
- **Possible values**: `true`, `false`
- **Stable**: No

#### `false` (default):

```rust
let lorem = "ipsum dolor sit amet consectetur adipiscing elit lorem ipsum dolor sit";
```

#### `true`:

```rust
let lorem =
    "ipsum dolor sit amet consectetur \
     adipiscing elit lorem ipsum dolor sit";
```

See also [`max_width`](#max_width).

## `hard_tabs`

Use tab characters for indentation, spaces for alignment

- **Default value**: `false`
- **Possible values**: `true`, `false`
- **Stable**: Yes

#### `false` (default):

```rust
fn lorem() -> usize {
    42 // spaces before 42
}
```

#### `true`:

```rust
fn lorem() -> usize {
	42 // tabs before 42
}
```

See also: [`tab_spaces`](#tab_spaces).


## `imports_indent`

Indent style of imports

- **Default Value**: `"Visual"`
- **Possible values**: `"Block"`, `"Visual"`
- **Stable**: No

#### `"Visual"` (default):

```rust
use foo::{xxx,
          yyy,
          zzz};
```

#### `"Block"`:

```rust
use foo::{
    xxx,
    yyy,
    zzz,
};
```

See also: [`imports_layout`](#imports_layout).

## `imports_layout`

Item layout inside a imports block

- **Default value**: "Mixed"
- **Possible values**: "Horizontal", "HorizontalVertical", "Mixed", "Vertical"
- **Stable**: No

#### `"Mixed"` (default):

```rust
use foo::{xxx, yyy, zzz};

use foo::{aaa, bbb, ccc,
          ddd, eee, fff};
```

#### `"Horizontal"`:

**Note**: This option forces all imports onto one line and may exceed `max_width`.

```rust
use foo::{xxx, yyy, zzz};

use foo::{aaa, bbb, ccc, ddd, eee, fff};
```

#### `"HorizontalVertical"`:

```rust
use foo::{xxx, yyy, zzz};

use foo::{aaa,
          bbb,
          ccc,
          ddd,
          eee,
          fff};
```

#### `"Vertical"`:

```rust
use foo::{xxx,
          yyy,
          zzz};

use foo::{aaa,
          bbb,
          ccc,
          ddd,
          eee,
          fff};
```


## `match_block_trailing_comma`

Put a trailing comma after a block based match arm (non-block arms are not affected)

- **Default value**: `false`
- **Possible values**: `true`, `false`
- **Stable**: No

#### `false` (default):

```rust
match lorem {
    Lorem::Ipsum => {
        println!("ipsum");
    }
    Lorem::Dolor => println!("dolor"),
}
```

#### `true`:

```rust
match lorem {
    Lorem::Ipsum => {
        println!("ipsum");
    },
    Lorem::Dolor => println!("dolor"),
}
```

See also: [`trailing_comma`](#trailing_comma), [`match_arm_blocks`](#match_arm_blocks).

## `max_width`

Maximum width of each line

- **Default value**: `100`
- **Possible values**: any positive integer
- **Stable**: Yes

See also [`error_on_line_overflow`](#error_on_line_overflow).

## `merge_derives`

Merge multiple derives into a single one.

- **Default value**: `true`
- **Possible values**: `true`, `false`
- **Stable**: Yes

#### `true` (default):

```rust
#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub enum Foo {}
```

#### `false`:

```rust
#[derive(Eq, PartialEq)]
#[derive(Debug)]
#[derive(Copy, Clone)]
pub enum Foo {}
```

## `force_multiline_blocks`

Force multiline closure and match arm bodies to be wrapped in a block

- **Default value**: `false`
- **Possible values**: `false`, `true`
- **Stable**: No

#### `false` (default):

```rust
result.and_then(|maybe_value| match maybe_value {
    None => ...,
    Some(value) => ...,
})

match lorem {
    None => if ipsum {
        println!("Hello World");
    },
    Some(dolor) => ...,
}
```

#### `true`:

```rust

result.and_then(|maybe_value| {
    match maybe_value {
        None => ...,
        Some(value) => ...,
    }
})

match lorem {
    None => {
        if ipsum {
            println!("Hello World");
        }
    }
    Some(dolor) => ...,
}
```


## `newline_style`

Unix or Windows line endings

- **Default value**: `"Unix"`
- **Possible values**: `"Native"`, `"Unix"`, `"Windows"`
- **Stable**: Yes

## `normalize_comments`

Convert /* */ comments to // comments where possible

- **Default value**: `false`
- **Possible values**: `true`, `false`
- **Stable**: Yes

#### `false` (default):

```rust
// Lorem ipsum:
fn dolor() -> usize {}

/* sit amet: */
fn adipiscing() -> usize {}
```

#### `true`:

```rust
// Lorem ipsum:
fn dolor() -> usize {}

// sit amet:
fn adipiscing() -> usize {}
```

## `reorder_imported_names`

Reorder lists of names in import statements alphabetically

- **Default value**: `false`
- **Possible values**: `true`, `false`
- **Stable**: No

#### `false` (default):

```rust
use super::{lorem, ipsum, dolor, sit};
```

#### `true`:

```rust
use super::{dolor, ipsum, lorem, sit};
```

See also [`reorder_imports`](#reorder_imports).

## `reorder_imports`

Reorder import statements alphabetically

- **Default value**: `false`
- **Possible values**: `true`, `false`
- **Stable**: No

#### `false` (default):

```rust
use lorem;
use ipsum;
use dolor;
use sit;
```

#### `true`:

```rust
use dolor;
use ipsum;
use lorem;
use sit;
```

See also [`reorder_imported_names`](#reorder_imported_names), [`reorder_imports_in_group`](#reorder_imports_in_group).

## `reorder_imports_in_group`

Reorder import statements in group

- **Default value**: `false`
- **Possible values**: `true`, `false`
- **Stable**: No

**Note:** This option takes effect only when [`reorder_imports`](#reorder_imports) is set to `true`.

#### `false` (default):

```rust
use std::mem;
use std::io;

use lorem;
use ipsum;
use dolor;
use sit;
```

#### `true`:

```rust
use std::io;
use std::mem;

use dolor;
use ipsum;
use lorem;
use sit;
```

See also [`reorder_imports`](#reorder_imports).

## `reorder_extern_crates`

Reorder `extern crate` statements alphabetically

- **Default value**: `true`
- **Possible values**: `true`, `false`
- **Stable**: No

#### `true` (default):

```rust
extern crate dolor;
extern crate ipsum;
extern crate lorem;
extern crate sit;
```

#### `false`:

```rust
extern crate lorem;
extern crate ipsum;
extern crate dolor;
extern crate sit;
```

See also [`reorder_extern_crates_in_group`](#reorder_extern_crates_in_group).

## `reorder_extern_crates_in_group`

Reorder `extern crate` statements in group

- **Default value**: `true`
- **Possible values**: `true`, `false`
- **Stable**: No

**Note:** This option takes effect only when [`reorder_imports`](#reorder_imports) is set to `true`.

#### `true` (default):

```rust
extern crate a;
extern crate b;

extern crate dolor;
extern crate ipsum;
extern crate lorem;
extern crate sit;
```

#### `false`:

```rust
extern crate b;
extern crate a;

extern crate lorem;
extern crate ipsum;
extern crate dolor;
extern crate sit;
```

See also [`reorder_extern_crates`](#reorder_extern_crates).

## `report_todo`

Report `TODO` items in comments.

- **Default value**: `"Never"`
- **Possible values**: `"Always"`, `"Unnumbered"`, `"Never"`
- **Stable**: No

Warns about any comments containing `TODO` in them when set to `"Always"`. If
it contains a `#X` (with `X` being a number) in parentheses following the
`TODO`, `"Unnumbered"` will ignore it.

See also [`report_fixme`](#report_fixme).

## `report_fixme`

Report `FIXME` items in comments.

- **Default value**: `"Never"`
- **Possible values**: `"Always"`, `"Unnumbered"`, `"Never"`
- **Stable**: No

Warns about any comments containing `FIXME` in them when set to `"Always"`. If
it contains a `#X` (with `X` being a number) in parentheses following the
`FIXME`, `"Unnumbered"` will ignore it.

See also [`report_todo`](#report_todo).


## `skip_children`

Don't reformat out of line modules

- **Default value**: `false`
- **Possible values**: `true`, `false`
- **Stable**: No

## `space_after_colon`

Leave a space after the colon.

- **Default value**: `true`
- **Possible values**: `true`, `false`
- **Stable**: No

#### `true` (default):

```rust
fn lorem<T: Eq>(t: T) {
    let lorem: Dolor = Lorem {
        ipsum: dolor,
        sit: amet,
    };
}
```

#### `false`:

```rust
fn lorem<T:Eq>(t:T) {
    let lorem:Dolor = Lorem {
        ipsum:dolor,
        sit:amet,
    };
}
```

See also: [`space_before_colon`](#space_before_colon).

## `space_before_colon`

Leave a space before the colon.

- **Default value**: `false`
- **Possible values**: `true`, `false`
- **Stable**: No

#### `false` (default):

```rust
fn lorem<T: Eq>(t: T) {
    let lorem: Dolor = Lorem {
        ipsum: dolor,
        sit: amet,
    };
}
```

#### `true`:

```rust
fn lorem<T : Eq>(t : T) {
    let lorem : Dolor = Lorem {
        ipsum : dolor,
        sit : amet,
    };
}
```

See also: [`space_after_colon`](#space_after_colon).

## `struct_field_align_threshold`

The maximum diff of width between struct fields to be aligned with each other.

- **Default value** : 0
- **Possible values**: any positive integer
- **Stable**: No

#### `0` (default):

```rust
struct Foo {
    x: u32,
    yy: u32,
    zzz: u32,
}
```

#### `20`:

```rust
struct Foo {
    x:   u32,
    yy:  u32,
    zzz: u32,
}
```

## `spaces_around_ranges`

Put spaces around the .. and ... range operators

- **Default value**: `false`
- **Possible values**: `true`, `false`
- **Stable**: No

#### `false` (default):

```rust
let lorem = 0..10;
```

#### `true`:

```rust
let lorem = 0 .. 10;
```

## `spaces_within_parens_and_brackets`

Put spaces within non-empty generic arguments, parentheses, and square brackets

- **Default value**: `false`
- **Possible values**: `true`, `false`
- **Stable**: No

#### `false` (default):

```rust
// generic arguments
fn lorem<T: Eq>(t: T) {
    // body
}

// non-empty parentheses
fn lorem<T: Eq>(t: T) {
    let lorem = (ipsum, dolor);
}

// non-empty square brackets
let lorem: [usize; 2] = [ipsum, dolor];
```

#### `true`:

```rust
// generic arguments
fn lorem< T: Eq >(t: T) {
    // body
}

// non-empty parentheses
fn lorem<T: Eq>( t: T ) {
    let lorem = ( ipsum, dolor );
}

// non-empty square brackets
let lorem: [ usize; 2 ] = [ ipsum, dolor ];
```

## `struct_lit_single_line`

Put small struct literals on a single line

- **Default value**: `true`
- **Possible values**: `true`, `false`
- **Stable**: No

#### `true` (default):

```rust
let lorem = Lorem { ipsum: dolor, sit: amet };
```

#### `false`:

```rust
let lorem = Lorem {
    ipsum: dolor,
    sit: amet,
};
```

See also: [`indent_style`](#indent_style).


## `tab_spaces`

Number of spaces per tab

- **Default value**: `4`
- **Possible values**: any positive integer
- **Stable**: Yes

#### `4` (default):

```rust
fn lorem() {
    let ipsum = dolor();
    let sit = vec![
        "amet consectetur adipiscing elit."
    ];
}
```

#### `2`:

```rust
fn lorem() {
  let ipsum = dolor();
  let sit = vec![
    "amet consectetur adipiscing elit."
  ];
}
```

See also: [`hard_tabs`](#hard_tabs).


## `trailing_comma`

How to handle trailing commas for lists

- **Default value**: `"Vertical"`
- **Possible values**: `"Always"`, `"Never"`, `"Vertical"`
- **Stable**: No

#### `"Vertical"` (default):

```rust
let Lorem { ipsum, dolor, sit } = amet;
let Lorem {
    ipsum,
    dolor,
    sit,
    amet,
    consectetur,
    adipiscing,
} = elit;
```

#### `"Always"`:

```rust
let Lorem { ipsum, dolor, sit, } = amet;
let Lorem {
    ipsum,
    dolor,
    sit,
    amet,
    consectetur,
    adipiscing,
} = elit;
```

#### `"Never"`:

```rust
let Lorem { ipsum, dolor, sit } = amet;
let Lorem {
    ipsum,
    dolor,
    sit,
    amet,
    consectetur,
    adipiscing
} = elit;
```

See also: [`match_block_trailing_comma`](#match_block_trailing_comma).

## `trailing_semicolon`

Add trailing semicolon after break, continue and return

- **Default value**: `true`
- **Possible values**: `true`, `false`
- **Stable**: No

#### `true` (default):
```rust
fn foo() -> usize {
    return 0;
}
```

#### `false`:
```rust
fn foo() -> usize {
    return 0
}
```

## `type_punctuation_density`

Determines if `+` or `=` are wrapped in spaces in the punctuation of types

- **Default value**: `"Wide"`
- **Possible values**: `"Compressed"`, `"Wide"`
- **Stable**: No

#### `"Wide"` (default):

```rust
fn lorem<Ipsum: Dolor + Sit = Amet>() {
	// body
}
```

#### `"Compressed"`:

```rust
fn lorem<Ipsum: Dolor+Sit=Amet>() {
	// body
}
```

## `use_try_shorthand`

Replace uses of the try! macro by the ? shorthand

- **Default value**: `false`
- **Possible values**: `true`, `false`
- **Stable**: No

#### `false` (default):

```rust
let lorem = try!(ipsum.map(|dolor|dolor.sit()));
```

#### `true`:

```rust
let lorem = ipsum.map(|dolor| dolor.sit())?;
```


## `wrap_comments`

Break comments to fit on the line

- **Default value**: `false`
- **Possible values**: `true`, `false`
- **Stable**: Yes

#### `false` (default):

```rust
// Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
```

#### `true`:

```rust
// Lorem ipsum dolor sit amet, consectetur adipiscing elit,
// sed do eiusmod tempor incididunt ut labore et dolore
// magna aliqua. Ut enim ad minim veniam, quis nostrud
// exercitation ullamco laboris nisi ut aliquip ex ea
// commodo consequat.
```

## `match_arm_blocks`

Wrap the body of arms in blocks when it does not fit on the same line with the pattern of arms

- **Default value**: `true`
- **Possible values**: `true`, `false`
- **Stable**: No

#### `true` (default):

```rust
match lorem {
    true => {
        foooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo(x)
    }
    false => println!("{}", sit),
}
```

#### `false`:

```rust
match lorem {
    true =>
        foooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo(x),
    false => println!("{}", sit),
}
```

See also: [`match_block_trailing_comma`](#match_block_trailing_comma).

## `write_mode`

What Write Mode to use when none is supplied: Replace, Overwrite, Display, Diff, Coverage

- **Default value**: `"Overwrite"`
- **Possible values**: `"Checkstyle"`, `"Coverage"`, `"Diff"`, `"Display"`, `"Overwrite"`, `"Plain"`, `"Replace"`
- **Stable**: No

## `blank_lines_upper_bound`

Maximum number of blank lines which can be put between items. If more than this number of consecutive empty
lines are found, they are trimmed down to match this integer.

- **Default value**: `1`
- **Possible values**: *unsigned integer*
- **Stable**: No

### Example
Original Code:

```rust
fn foo() {
    println!("a");
}



fn bar() {
    println!("b");


    println!("c");
}
```

#### `1` (default):
```rust
fn foo() {
    println!("a");
}

fn bar() {
    println!("b");

    println!("c");
}
```

#### `2` (default):
```rust
fn foo() {
    println!("a");
}


fn bar() {
    println!("b");


    println!("c");
}
```

See also: [`blank_lines_lower_bound`](#blank_lines_lower_bound)

## `blank_lines_lower_bound`

Minimum number of blank lines which must be put between items. If two items have fewer blank lines between
them, additional blank lines are inserted.

- **Default value**: `0`
- **Possible values**: *unsigned integer*
- **Stable**: No

### Example
Original Code (rustfmt will not change it with the default value of `0`):

```rust
fn foo() {
    println!("a");
}
fn bar() {
    println!("b");
    println!("c");
}
```

#### `1`
```rust
fn foo() {

    println!("a");
}

fn bar() {

    println!("b");

    println!("c");
}
```
