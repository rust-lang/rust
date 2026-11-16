//@ rustfmt-remove_nested_parens: true

#![feature(stmt_expr_attributes)]


fn foo() -> u8 {
    // No outer parens
    {
    #[allow(arithmetic_overflow)]
    (1u8 + 255u8)
    }

    // One level of parens
    {
    (
        #[allow(arithmetic_overflow)]
        (1u8 + 255u8)
    )
    }

    // One level of parens; in-between line comment
    {
    (
        #[allow(arithmetic_overflow)]
        // in-between comment
        (1u8 + 255u8)
    )
    }

    // One level of parens; in-between inline comment
    {
    (
        #[allow(arithmetic_overflow)]
        /* in-between comment */
        (1u8 + 255u8)
    )
    }

    // One level of parens; pre comment line comment
    {
    (
        // pre comment
        #[allow(arithmetic_overflow)]
        (1u8 + 255u8)
    )
    }

    // One level of parens; pre comment inline comment
    {
    (
        /* pre comment */
        #[allow(arithmetic_overflow)]
        (1u8 + 255u8)
    )
    }

    // One level of parens; post comment line comment
    {
    (
        #[allow(arithmetic_overflow)]
        (1u8 + 255u8)
        // post comment
    )
    }

    // One level of parens; post comment inline comment
    {
    (
        #[allow(arithmetic_overflow)]
        (1u8 + 255u8)
        /* post comment */
    )
    }

    // One level of parens; pre, in-between, and post line comments
    {
    (
        // pre comment
        #[allow(arithmetic_overflow)]
        // in-between comment
        (1u8 + 255u8)
        // post comment
    )
    }

    // One level of parens; pre, in-between, and post inline comments
    {
    (
        /* pre comment */
        #[allow(arithmetic_overflow)]
        /* in-between comment */
        (1u8 + 255u8)
        /* post comment */
    )
    }

    // One level of parens; pre, in-between, and post mix line and inline comments
    {
    (
        /* pre comment */
        #[allow(arithmetic_overflow)]
        // in-between comment
        (1u8 + 255u8)
        /* post comment */
    )
    }

    // One level of parens; pre, in-between, and post mix line and inline comments
    {
    (
        // pre comment
        #[allow(arithmetic_overflow)]
        /* in-between comment */
        (1u8 + 255u8)
        // post comment
    )
    }
}


fn nested_parens() -> u8 {
    // Deeply nested parens
    {
    ((((
        #[allow(arithmetic_overflow)]
        (1u8 + 255u8)
    ))))
    }

    // Deeply nested parens; in-between line comment
    {
    ((((
        #[allow(arithmetic_overflow)]
        // in-between comment
        (1u8 + 255u8)
    ))))
    }

    // Deeply nested parens; in-between inline comment
    {
    ((((
        #[allow(arithmetic_overflow)]
        /* in-between comment */
        (1u8 + 255u8)
    ))))
    }

    // Deeply nested parens; pre comment line comment
    {
    ((((
        // pre comment
        #[allow(arithmetic_overflow)]
        (1u8 + 255u8)
    ))))
    }

    // Deeply nested parens; pre comment inline comment
    {
    ((((
        /* pre comment */
        #[allow(arithmetic_overflow)]
        (1u8 + 255u8)
    ))))
    }

    // Deeply nested parens; post comment line comment
    {
    ((((
        #[allow(arithmetic_overflow)]
        (1u8 + 255u8)
        // post comment
    ))))
    }

    // Deeply nested parens; post comment inline comment
    {
    ((((
        #[allow(arithmetic_overflow)]
        (1u8 + 255u8)
        /* post comment */
    ))))
    }

    // Deeply nested parens; pre, in-between, and post line comments
    {
    ((((
        // pre comment
        #[allow(arithmetic_overflow)]
        // in-between comment
        (1u8 + 255u8)
        // post comment
    ))))
    }

    // Deeply nested parens; pre, in-between, and post inline comments
    {
    ((((
        /* pre comment */
        #[allow(arithmetic_overflow)]
        /* in-between comment */
        (1u8 + 255u8)
        /* post comment */
    ))))
    }

    // Deeply nested parens; pre, in-between, and post mix line and inline comments
    {
    ((((
        /* pre comment */
        #[allow(arithmetic_overflow)]
        // in-between comment
        (1u8 + 255u8)
        /* post comment */
    ))))
    }

    // Deeply nested parens; pre, in-between, and post mix line and inline comments
    {
    ((((
        // pre comment
        #[allow(arithmetic_overflow)]
        /* in-between comment */
        (1u8 + 255u8)
        // post comment
    ))))
    }


    // Deeply nested parens before and after
    {
    ((((
        #[allow(arithmetic_overflow)]
        (((((1u8 + 255u8)))))
    ))))
    }

    // Deeply nested parens; in-between line comment
    {
    ((((
        #[allow(arithmetic_overflow)]
        // in-between comment
        ((((((1u8 + 255u8))))))
    ))))
    }

    // Deeply nested parens; in-between inline comment
    {
    ((((
        #[allow(arithmetic_overflow)]
        /* in-between comment */
        ((((((1u8 + 255u8))))))
    ))))
    }

    // Deeply nested parens; pre comment line comment
    {
    ((((
        // pre comment
        #[allow(arithmetic_overflow)]
        ((((((1u8 + 255u8))))))
    ))))
    }

    // Deeply nested parens; pre comment inline comment
    {
    ((((
        /* pre comment */
        #[allow(arithmetic_overflow)]
        ((((((1u8 + 255u8))))))
    ))))
    }

    // Deeply nested parens; post comment line comment
    {
    ((((
        #[allow(arithmetic_overflow)]
        ((((((1u8 + 255u8))))))
        // post comment
    ))))
    }

    // Deeply nested parens; post comment inline comment
    {
    ((((
        #[allow(arithmetic_overflow)]
        ((((((1u8 + 255u8))))))
        /* post comment */
    ))))
    }

    // Deeply nested parens; pre, in-between, and post line comments
    {
    ((((
        // pre comment
        #[allow(arithmetic_overflow)]
        // in-between comment
        ((((((1u8 + 255u8))))))
        // post comment
    ))))
    }

    // Deeply nested parens; pre, in-between, and post inline comments
    {
    ((((
        /* pre comment */
        #[allow(arithmetic_overflow)]
        /* in-between comment */
        ((((((1u8 + 255u8))))))
        /* post comment */
    ))))
    }

    // Deeply nested parens; pre, in-between, and post mix line and inline comments
    {
    ((((
        /* pre comment */
        #[allow(arithmetic_overflow)]
        // in-between comment
        ((((((1u8 + 255u8))))))
        /* post comment */
    ))))
    }

    // Deeply nested parens; pre, in-between, and post mix line and inline comments
    {
    ((((
        // pre comment
        #[allow(arithmetic_overflow)]
        /* in-between comment */
        ((((((1u8 + 255u8))))))
        // post comment
    ))))
    }
}

// Testing the pre_comment detection with inner attrs
fn inner_attrs() -> u8 {
    {
        (
            // pre comment
            {
                #![allow(unused)]
                1
            }
        )
    }

    {
        (/* pre comment */{
            #![allow(unused)]
            1
        })
    }
}

// Testing the pre_comment detection with multiple outer attrs
fn multiple_outer_attrs() -> u8 {
    {
    (
        // pre comment
        #[attr_a]
        #[attr_b]
        #[attr_c]
        (1)
    )
    }

    {
    (
        // pre comment
        #[attr_a]
        // Some inner comment
        #[attr_b]
        #[attr_c]
        (1)
    )
    }

    {
    (
        // pre comment
        #[attr_a]
        // Some inner comment
        #[attr_b]
        // Another inner comment
        #[attr_c]
        (1)
    )
    }
}
