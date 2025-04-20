// Regression test for #132727
//
// This ensures we correctly handle nested replacement ranges in cfg_eval.
// PR #129346 included a code simplification in collect_tokens that
// caused a regression with nested cfg attributes, which was reverted in #132587.

//@ check-pass

#![feature(cfg_eval)]
#![feature(stmt_expr_attributes)]

fn nested_cfg_attributes() -> u32 {
    // This test reproduces the core issue: nested cfg replacement ranges
    // The outer #[cfg_eval] processes the inner #[cfg] attribute,
    // creating a nested replacement range situation
    #[cfg_eval] #[cfg(not(FALSE))] 0
}

// Another test case with more complex nesting
fn multi_level_nesting() -> u32 {
    let result = {
        #[cfg_eval]
        {
            #[cfg(not(FALSE))]
            {
                #[cfg(not(FALSE))]
                42
            }
        }
    };
    result
}

// Test for overlapping nested cfg attributes with different conditions
fn overlapping_cfg_attributes() -> u32 {
    // This tests a more complex case where the cfg attributes have different conditions
    // and are deeply nested, which could potentially trigger issues with replacement ranges
    #[cfg_eval]
    {
        #[cfg(any(not(FALSE), FALSE))]
        {
            #[cfg(all(not(FALSE), not(FALSE)))]
            {
                #[cfg(not(FALSE))]
                100
            }
        }
    }
}

// Test for the interaction between cfg_eval and cfg_attr
fn cfg_eval_with_cfg_attr() -> u32 {
    // This tests the interaction between cfg_eval and cfg_attr, which was
    // one of the main issues addressed in the regression fix
    #[cfg_eval]
    #[cfg_attr(not(FALSE), cfg_attr(not(FALSE), cfg(not(FALSE))))]
    200
}

fn main() {
    assert_eq!(nested_cfg_attributes(), 0);
    assert_eq!(multi_level_nesting(), 42);
    assert_eq!(overlapping_cfg_attributes(), 100);
    assert_eq!(cfg_eval_with_cfg_attr(), 200);
}
