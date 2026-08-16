// rustfmt-stable: true

// While we gate the `cfg_select!` formatting behind the `is_nightly_channel!()` check
// this test helps ensure that we don't start formatting `cfg_select!` on the `stable`
// or `beta` release channels. It is intentionally formatted incorrectly. As soon as the
// `is_nightly_channel!()` gate is removed this will start formatting and we can remove this test.
cfg_select! (
       unix     => 1,
       windows     => 1,
);
