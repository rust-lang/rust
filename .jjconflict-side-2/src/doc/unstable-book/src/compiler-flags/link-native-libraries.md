# `link-native-libraries`

This option allows ignoring libraries specified in `#[link]` attributes instead of passing them to the linker.
This can be useful in build systems that manage native libraries themselves and pass them manually,
e.g. with `-Clink-arg`.

- `yes` - Pass native libraries to the linker. Default.
- `no` - Don't pass native libraries to the linker.
