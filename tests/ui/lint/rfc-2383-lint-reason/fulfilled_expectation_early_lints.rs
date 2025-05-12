//@ check-pass

fn expect_early_pass_lints() {
    #[expect(while_true)]
    while true {
        println!("I never stop")
    }

    #[expect(unused_doc_comments)]
    /// This comment triggers the `unused_doc_comments` lint
    let _sheep = "wolf";

    let x = 123;
    #[expect(ellipsis_inclusive_range_patterns)]
    match x {
        0...100 => {}
        _ => {}
    }
}

fn main() {}
