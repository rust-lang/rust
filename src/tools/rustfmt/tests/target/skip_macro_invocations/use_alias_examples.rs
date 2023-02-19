// rustfmt-skip_macro_invocations: ["aaa","ccc"]

// These tests demonstrate a realistic use case with use aliases.
// The use statements should not impact functionality in any way.

use crate::{aaa, bbb, ddd};

// No use alias, invocation in list
// Should skip this invocation
aaa!(
        const _: u8 = 0;
);

// Use alias, invocation in list
// Should skip this invocation
use crate::bbb as ccc;
ccc!(
        const _: u8 = 0;
);

// Use alias, invocation not in list
// Should not skip this invocation
use crate::ddd as eee;
eee!(
    const _: u8 = 0;
);

// No use alias, invocation not in list
// Should not skip this invocation
fff!(
    const _: u8 = 0;
);
