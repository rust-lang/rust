fn by_ref_with_block_before_ident() {
    if let Some(ref /*def*/ state) = foo {
        println!("asdfasdfasdf");
    }
}

fn mut_block_before_ident() {
    if let Some(mut /*def*/ state) = foo {
        println!("123");
    }
}

fn ref_and_mut_blocks_before_ident() {
    if let Some(ref /*abc*/ mut /*def*/ state) = foo {
        println!("deefefefefefwea");
    }
}

fn sub_pattern() {
    let foo @ /*foo*/ bar(f) = 42;
}

fn no_prefix_block_before_ident() {
    if let Some(/*def*/ state) = foo {
        println!("129387123123");
    }
}

fn issue_3853() {
    if let Some(ref /*mut*/ state) = foo {}
}

fn double_slash_comment_between_lhs_and_rhs() {
    if let Some(e) =
        // self.foo.bar(e, tx)
        packet.transaction.state.committed
    {
        // body
        println!("a2304712836123");
    }
}

fn block_comment_between_lhs_and_rhs() {
    if let Some(ref /*def*/ mut /*abc*/ state) = /*abc*/ foo {
        println!("asdfasdfasdf");
    }
}
