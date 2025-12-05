# Changelog #256

Hello!

Commit: [`0123456`](https://github.com/rust-lang/rust-analyzer/commit/0123456789abcdef0123456789abcdef01234567) \
Release: [`2022-01-01`](https://github.com/rust-lang/rust-analyzer/releases/2022-01-01)

## New Features

- **BREAKING** [`#1111`](https://github.com/rust-lang/rust-analyzer/pull/1111) shortcut <kbd>ctrl</kbd>+<kbd>r</kbd>
  - hyphen-prefixed list item
- nested list item
  - `foo` -> `foofoo`
  - `bar` -> `barbar`
- listing in the secondary level
  1. install
  1. add to config

     ```json
     {"foo":"bar"}
     ```
- list item with continuation

  ![](https://example.com/animation.gif)

  ![alt text](https://example.com/animation.gif)

  <video src="https://example.com/movie.mp4" controls loop>Your browser does not support the video tag.</video>

  <video src="https://example.com/movie.mp4" autoplay controls loop>Your browser does not support the video tag.</video>

  _Image_\
  ![](https://example.com/animation.gif)

  _Video_\
  <video src="https://example.com/movie.mp4" controls loop>Your browser does not support the video tag.</video>

  ```bash
  rustup update nightly
  ```

  ```
  This is a plain listing.
  ```
- single line item followed by empty lines
- multiline list
  item followed by empty lines
- multiline list
  item with indent
- multiline list
  item not followed by empty lines
- multiline list
  item followed by different marker
  - foo
  - bar
- multiline list
  item followed by list continuation

  paragraph
  paragraph

## Another Section

- foo bar baz
- list item with an inline image
  ![](https://example.com/animation.gif)

The highlight of the month is probably [`#1111`](https://github.com/rust-lang/rust-analyzer/pull/1111).
See [online manual](https://example.com/manual) for more information.

```bash
rustup update nightly
```

```
rustup update nightly
```

```
This is a plain listing.
```
