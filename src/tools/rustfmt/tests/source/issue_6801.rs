// rustfmt-wrap_comments: true
// rustfmt-comment_width: 80
// rustfmt-max_width: 200

fn foo() {
    // In this line, the next '.' is exactly at'comment_width'                 . It should break on that dot

    // In this line, 'comment_width' is reached in the middle of ThisVeryLongWord. It should break before that word

    {
        {
            {
                {
                    // again 'comment_width' is just at the next '.'           . Here's some more stuff
                    fn f1() {
                        fn f2() {
                            fn f3() {
                                fn f4() {
                                    fn f5() {
                                        fn f6() {
                                            fn f7() {
                                                fn f8() {
                                                    fn f9() {
                                                        fn f10() {
                                                            fn f11() {
                                                                // again       . Deeply nested comment
                                                                fn f12() {
                                                                    fn f13() {
                                                                        fn f14() {
                                                                            fn f15() {
                                                                                // indentation means this comment starts after 'comment_width'
                                                                                // we don't touch this comment
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
