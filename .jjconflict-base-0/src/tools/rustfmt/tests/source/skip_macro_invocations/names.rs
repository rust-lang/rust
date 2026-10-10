// rustfmt-skip_macro_invocations: ["foo","bar"]

// Should skip this invocation
foo!(
        const _: u8 = 0;
);

// Should skip this invocation
bar!(
        const _: u8 = 0;
);

// Should not skip this invocation
baz!(
        const _: u8 = 0;
);
