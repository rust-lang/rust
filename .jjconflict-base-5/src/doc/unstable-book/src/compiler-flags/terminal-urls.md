# `-Z terminal-urls`

The tracking feature for this issue is [#125586]

[#125586]: https://github.com/rust-lang/rust/issues/125586

---

This flag takes either a boolean or the string "auto".

When enabled, use the OSC 8 hyperlink terminal specification to print hyperlinks in the compiler output.
Use "auto" to try and autodetect whether the terminal emulator supports hyperlinks.
Currently, "auto" only enables hyperlinks if `COLORTERM=truecolor` and `TERM=xterm-256color`.
