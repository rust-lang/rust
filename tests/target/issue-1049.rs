// Test overlong function signature
pub unsafe fn reborrow_mut(
    &mut X: Abcde,
) -> Handle<NodeRef<marker::Mut, K, V, NodeType>, HandleType> {
}

pub fn merge(
    mut X: Abcdef,
) -> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Internal>, marker::Edge> {
}

impl Handle {
    pub fn merge(
        a: Abcd,
    ) -> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Internal>, marker::Edge> {
    }
}

// Long function without return type that should not be reformatted.
fn veeeeeeeeeeeeeeeeeeeeery_long_name(a: FirstTypeeeeeeeeee, b: SecondTypeeeeeeeeeeeeeeeeeeeeeee) {}

fn veeeeeeeeeeeeeeeeeeeeeery_long_name(a: FirstTypeeeeeeeeee, b: SecondTypeeeeeeeeeeeeeeeeeeeeeee) {
}

fn veeeeeeeeeeeeeeeeeeeeeeery_long_name(
    a: FirstTypeeeeeeeeee,
    b: SecondTypeeeeeeeeeeeeeeeeeeeeeee,
) {
}
