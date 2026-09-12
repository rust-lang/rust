//! Regression test for https://github.com/rust-lang/rust/issues/72323.
//! Hard block expectations constrain locals used as direct tail expressions.

//@ check-pass

fn empty_vec() -> impl Iterator<Item = i32> {
    std::iter::empty()
}

fn expected_from_local_type() {
    let nums: Vec<i32> = {
        let mut values = empty_vec().collect();
        values.sort();
        values
    };

    assert!(nums.is_empty());
}

fn nested_block_constraints_are_scoped() {
    let outer: Vec<i32> = {
        let inner: Vec<i32> = {
            let mut values = empty_vec().collect();
            values.sort();
            values
        };
        assert!(inner.is_empty());

        let mut values = empty_vec().collect();
        values.sort();
        values
    };

    assert!(outer.is_empty());
}

fn takes_vec(_: Vec<i32>) {}

fn expected_from_function_argument() {
    takes_vec({
        let mut values = empty_vec().collect();
        values.sort();
        values
    });
}

fn expected_from_match() {
    let _: Vec<i32> = match true {
        true => {
            let mut values = empty_vec().collect();
            values.sort();
            values
        }
        false => Vec::new(),
    };
}

fn labeled_block_with_break() {
    let _: Vec<i32> = 'block: {
        if false {
            break 'block Vec::new();
        }
        let mut values = empty_vec().collect();
        values.sort();
        values
    };
}

fn main() {
    expected_from_local_type();
    nested_block_constraints_are_scoped();
    expected_from_function_argument();
    expected_from_match();
    labeled_block_with_break();
}
