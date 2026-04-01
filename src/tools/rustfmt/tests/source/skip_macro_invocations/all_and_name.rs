// rustfmt-skip_macro_invocations: ["*","items"]

// Should skip this invocation
items!(
        const _: u8 = 0;
);

// Should also skip this invocation, as the wildcard covers it
renamed_items!(
        const _: u8 = 0;
);
