# `doc_cfg`

The tracking issue for this feature is: [#43781]

------

The `doc_cfg` feature allows an API be documented as only available in some specific platforms.
This attribute has two effects:

1. In the annotated item's documentation, there will be a message saying "This is supported on
    (platform) only".

2. The item's doc-tests will only run on the specific platform.

In addition to allowing the use of the `#[doc(cfg)]` attribute, this feature enables the use of a
special conditional compilation flag, `#[cfg(rustdoc)]`, set whenever building documentation on your
crate.

This feature was introduced as part of PR [#43348] to allow the platform-specific parts of the
standard library be documented.

```rust
#![feature(doc_cfg)]

#[cfg(any(windows, rustdoc))]
#[doc(cfg(windows))]
/// The application's icon in the notification area (a.k.a. system tray).
///
/// # Examples
///
/// ```no_run
/// extern crate my_awesome_ui_library;
/// use my_awesome_ui_library::current_app;
/// use my_awesome_ui_library::windows::notification;
///
/// let icon = current_app().get::<notification::Icon>();
/// icon.show();
/// icon.show_message("Hello");
/// ```
pub struct Icon {
    // ...
}
```

[#43781]: https://github.com/rust-lang/rust/issues/43781
[#43348]: https://github.com/rust-lang/rust/issues/43348
