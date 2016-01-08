// Match expressions.

fn foo() {
    // A match expression.
    match x {
        // Some comment.
        a => foo(),
        b if 0 < 42 => foo(),
        c => {
            // Another comment.
            // Comment.
            an_expression;
            foo()
        }
        Foo(ref bar) => {
            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
        }
        Pattern1 | Pattern2 | Pattern3 => false,
        Paternnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn |
        Paternnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn => blah,
        Patternnnnnnnnnnnnnnnnnnn |
        Patternnnnnnnnnnnnnnnnnnn |
        Patternnnnnnnnnnnnnnnnnnn |
        Patternnnnnnnnnnnnnnnnnnn => meh,

        Patternnnnnnnnnnnnnnnnnnn |
        Patternnnnnnnnnnnnnnnnnnn if looooooooooooooooooong_guard => meh,

        Patternnnnnnnnnnnnnnnnnnnnnnnnn |
        Patternnnnnnnnnnnnnnnnnnnnnnnnn if looooooooooooooooooooooooooooooooooooooooong_guard => {
            meh
        }

        // Test that earlier patterns can take the guard space
        (aaaa,
         bbbbb,
         ccccccc,
         aaaaa,
         bbbbbbbb,
         cccccc,
         aaaa,
         bbbbbbbb,
         cccccc,
         dddddd) |
        Patternnnnnnnnnnnnnnnnnnnnnnnnn if loooooooooooooooooooooooooooooooooooooooooong_guard => {}

        _ => {}
        ast::PathParameters::AngleBracketedParameters(ref data) if data.lifetimes.len() > 0 ||
                                                                   data.types.len() > 0 ||
                                                                   data.bindings.len() > 0 => {}
    }

    let whatever = match something {
        /// DOC COMMENT!
        Some(_) => 42,
        // Comment on an attribute.
        #[an_attribute]
        // Comment after an attribute.
        None => 0,
        #[rustfmt_skip]
        Blurb     =>     {                  }
    };
}

// Test that a match on an overflow line is laid out properly.
fn main() {
    let sub_span =
        match xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx {
            Some(sub_span) => Some(sub_span),
            None => sub_span,
        };
}

// Test that one-line bodies align.
fn main() {
    match r {
        Variableeeeeeeeeeeeeeeeee => {
            ("variable",
             vec!["id", "name", "qualname", "value", "type", "scopeid"],
             true,
             true)
        }
        Enummmmmmmmmmmmmmmmmmmmm => {
            ("enum",
             vec!["id", "qualname", "scopeid", "value"],
             true,
             true)
        }
        Variantttttttttttttttttttttttt => {
            ("variant",
             vec!["id", "name", "qualname", "type", "value", "scopeid"],
             true,
             true)
        }
    };

    match x {
        y => {
            // Block with comment. Preserve me.
        }
        z => {
            stmt();
        }
    }
}

fn matches() {
    match 1 {
        -1 => 10,
        1 => 1, // foo
        2 => 2,
        // bar
        3 => 3,
        _ => 0, // baz
    }
}

fn match_skip() {
    let _ = match Some(1) {
        #[rustfmt_skip]
        Some( n ) => n,
        None => 1,
    };
}

fn issue339() {
    match a {
        b => {}
        c => {}
        d => {}
        e => {}
        // collapsing here is safe
        ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff => {}
        // collapsing here exceeds line length
        ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffg => {
        }
        h => {
            // comment above block
        }
        i => {} // comment below block
        j => {
            // comment inside block
        }
        j2 => {
            // comments inside...
        } // ... and after
        // TODO uncomment when vertical whitespace is handled better
        // k => {
        //
        //     // comment with WS above
        // }
        // l => {
        //     // comment with ws below
        //
        // }
        m => {}
        n => {}
        o => {}
        p => {
            // Dont collapse me
        }
        q => {}
        r => {}
        s => 0, // s comment
        // t comment
        t => 1,
        u => 2,
        // TODO uncomment when block-support exists
        // v => {
        // } /* funky block
        //    * comment */
        // final comment
    }
}

fn issue355() {
    match mac {
        a => println!("a", b),
        b => vec![1, 2],
        c => vec!(3; 4),
        d => println!("a", b),
        e => vec![1, 2],
        f => vec!(3; 4),
        h => println!("a", b), // h comment
        i => vec![1, 2], // i comment
        j => vec!(3; 4), // j comment
        // k comment
        k => println!("a", b),
        // l comment
        l => vec![1, 2],
        // m comment
        m => vec!(3; 4),
        // Rewrite splits macro
        nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn => {
            println!("a", b)
        }
        // Rewrite splits macro
        oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo => {
            vec![1, 2]
        }
        // Macro support fails to recognise this macro as splitable
        // We push the whole expr to a new line, TODO split this macro as well
        pppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppp => {
            vec!(3; 4)
        }
        // q, r and s: Rewrite splits match arm
        qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq => {
            println!("a", b)
        }
        rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr => {
            vec![1, 2]
        }
        ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss => {
            vec!(3; 4)
        }
        // Funky bracketing styles
        t => println!{"a", b},
        u => vec![1, 2],
        v => vec!{3; 4},
        w => println!["a", b],
        x => vec![1, 2],
        y => vec![3; 4],
        // Brackets with comments
        tc => println!{"a", b}, // comment
        uc => vec![1, 2], // comment
        vc => vec!{3; 4}, // comment
        wc => println!["a", b], // comment
        xc => vec![1, 2], // comment
        yc => vec![3; 4], // comment
        yd => {
            looooooooooooooooooooooooooooooooooooooooooooooooooooooooong_func(aaaaaaaaaa,
                                                                              bbbbbbbbbb,
                                                                              cccccccccc,
                                                                              dddddddddd)
        }
    }
}

fn issue280() {
    {
        match x {
            CompressionMode::DiscardNewline | CompressionMode::CompressWhitespaceNewline => {
                ch == '\n'
            }
            ast::ItemConst(ref typ, ref expr) => {
                self.process_static_or_const_item(item, &typ, &expr)
            }
        }
    }
}

fn issue383() {
    match resolution.last_private {
        LastImport{..} => false,
        _ => true,
    };
}

fn issue507() {
    match 1 {
        1 => unsafe { std::intrinsics::abort() },
        _ => (),
    }
}

fn issue508() {
    match s.type_id() {
        Some(NodeTypeId::Element(ElementTypeId::HTMLElement(
                    HTMLElementTypeId::HTMLCanvasElement))) => true,
        Some(NodeTypeId::Element(ElementTypeId::HTMLElement(
                        HTMLElementTypeId::HTMLObjectElement))) => s.has_object_data(),
        Some(NodeTypeId::Element(_)) => false,
    }
}

fn issue496() {
    {
        {
            {
                match def {
                    def::DefConst(def_id) | def::DefAssociatedConst(def_id) => {
                        match const_eval::lookup_const_by_id(cx.tcx, def_id, Some(self.pat.id)) {
                            Some(const_expr) => x,
                        }
                    }
                }
            }
        }
    }
}

fn issue494() {
    {
        match stmt.node {
            hir::StmtExpr(ref expr, id) | hir::StmtSemi(ref expr, id) => {
                result.push(StmtRef::Mirror(Box::new(Stmt {
                    span: stmt.span,
                    kind: StmtKind::Expr {
                        scope: cx.tcx.region_maps.node_extent(id),
                        expr: expr.to_ref(),
                    },
                })))
            }
        }
    }
}
