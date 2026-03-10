// _0: in-macro
// _1: last argument in function invocation
// _2: non-last argument in function invocation
// _3: simple expression

// test0: the cause reported in issue: label is used, and there is usage, multiple statements
pub fn rustfmt_test0_0(condition: bool) {
    test_macro!(|transaction| 'block: {
        if condition {
            break 'block 0;
        }

        todo!()
    });
}

pub fn rustfmt_test0_1(condition: bool) {
    test_func(|transaction| 'block: {
        if condition {
            break 'block 0;
        }

        todo!()
    });
}

pub fn rustfmt_test0_2(condition: bool) {
    test_func2(
        |transaction| 'block: {
            if condition {
                break 'block 0;
            }

            todo!()
        },
        0,
    );
}

pub fn rustfmt_test0_3(condition: bool) {
    let x = |transaction| 'block: {
        if condition {
            break 'block 0;
        }

        todo!()
    };
}

// test1: label is unused, and there is usage, multiple statements
pub fn rustfmt_test1_0(condition: bool) {
    test_macro!(|transaction| 'block: {
        if condition {
            todo!("");
        }

        todo!()
    });
}

pub fn rustfmt_test1_1(condition: bool) {
    test_func(|transaction| 'block: {
        if condition {
            todo!("");
        }

        todo!()
    });
}

pub fn rustfmt_test1_2(condition: bool) {
    test_func2(
        |transaction| 'block: {
            if condition {
                todo!("");
            }

            todo!()
        },
        0,
    );
}

pub fn rustfmt_test1_3(condition: bool) {
    let x = |transaction| 'block: {
        if condition {
            todo!("");
        }

        todo!()
    };
}

// test2: label is used, single expression
pub fn rustfmt_test2_0(condition: bool) {
    test_macro!(|transaction| 'block: {
        break 'block 0;
    });
}

pub fn rustfmt_test2_1(condition: bool) {
    test_func(|transaction| 'block: {
        break 'block 0;
    });
}

pub fn rustfmt_test2_2(condition: bool) {
    test_func2(
        |transaction| 'block: {
            break 'block 0;
        },
        0,
    );
}

pub fn rustfmt_test2_3(condition: bool) {
    let x = |transaction| 'block: {
        break 'block 0;
    };
}

// test3: label is unused, single general multi-line expression
pub fn rustfmt_test3_0(condition: bool) {
    test_macro!(|transaction| 'block: {
        vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0,
        ]
    });
}

pub fn rustfmt_test3_1(condition: bool) {
    test_func(|transaction| 'block: {
        vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0,
        ]
    });
}

pub fn rustfmt_test3_2(condition: bool) {
    test_func2(
        |transaction| 'block: {
            vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0,
            ]
        },
        0,
    );
}

pub fn rustfmt_test3_3(condition: bool) {
    let x = |transaction| 'block: {
        vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0,
        ]
    };
}

// test4: label is unused, single block statement-expression
pub fn rustfmt_test4_0(condition: bool) {
    test_macro!(|transaction| 'block: {
        if condition {
            break 'block 1;
        } else {
            break 'block 0;
        }
    });
}

pub fn rustfmt_test4_1(condition: bool) {
    test_func(|transaction| 'block: {
        if condition {
            break 'block 1;
        } else {
            break 'block 0;
        }
    });
}

pub fn rustfmt_test4_2(condition: bool) {
    test_func2(
        |transaction| 'block: {
            if condition {
                break 'block 1;
            } else {
                break 'block 0;
            }
        },
        1,
    );
}

pub fn rustfmt_test4_3(condition: bool) {
    let x = |transaction| 'block: {
        if condition {
            break 'block 1;
        } else {
            break 'block 0;
        }
    };
}
