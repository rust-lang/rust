Each of these needs to be in a separate file,
because the `span_delayed_bug` ICE in rustdoc won't be triggered
if even a single other error was emitted.

However, conceptually they are all testing basically the same thing.
See https://github.com/rust-lang/rust/pull/73566#issuecomment-653689128
for more details.
