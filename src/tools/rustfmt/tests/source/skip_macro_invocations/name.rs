// rustfmt-skip_macro_invocations: ["items"]

// Should skip this invocation
items!(
        const _: u8 = 0;
);

// Should not skip this invocation
renamed_items!(
        const _: u8 = 0;
);
