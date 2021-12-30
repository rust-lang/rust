// rustfmt-normalize_comments: true
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
        Paternnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn
        | Paternnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn => blah,
        Patternnnnnnnnnnnnnnnnnnn
        | Patternnnnnnnnnnnnnnnnnnn
        | Patternnnnnnnnnnnnnnnnnnn
        | Patternnnnnnnnnnnnnnnnnnn => meh,

        Patternnnnnnnnnnnnnnnnnnn | Patternnnnnnnnnnnnnnnnnnn if looooooooooooooooooong_guard => {
            meh
        }

        Patternnnnnnnnnnnnnnnnnnnnnnnnn | Patternnnnnnnnnnnnnnnnnnnnnnnnn
            if looooooooooooooooooooooooooooooooooooooooong_guard =>
        {
            meh
        }

        // Test that earlier patterns can take the guard space
        (aaaa, bbbbb, ccccccc, aaaaa, bbbbbbbb, cccccc, aaaa, bbbbbbbb, cccccc, dddddd)
        | Patternnnnnnnnnnnnnnnnnnnnnnnnn
            if loooooooooooooooooooooooooooooooooooooooooong_guard => {}

        _ => {}
        ast::PathParameters::AngleBracketedParameters(ref data)
            if data.lifetimes.len() > 0 || data.types.len() > 0 || data.bindings.len() > 0 => {}
    }

    let whatever = match something {
        /// DOC COMMENT!
        Some(_) => 42,
        // Comment on an attribute.
        #[an_attribute]
        // Comment after an attribute.
        None => 0,
        #[rustfmt::skip]
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
        Variableeeeeeeeeeeeeeeeee => (
            "variable",
            vec!["id", "name", "qualname", "value", "type", "scopeid"],
            true,
            true,
        ),
        Enummmmmmmmmmmmmmmmmmmmm => (
            "enum",
            vec!["id", "qualname", "scopeid", "value"],
            true,
            true,
        ),
        Variantttttttttttttttttttttttt => (
            "variant",
            vec!["id", "name", "qualname", "type", "value", "scopeid"],
            true,
            true,
        ),
    };

    match x {
        y => { /*Block with comment. Preserve me.*/ }
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
        #[rustfmt::skip]
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
        h => { // comment above block
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
        p => { // Don't collapse me
        }
        q => {}
        r => {}
        s => 0, // s comment
        // t comment
        t => 1,
        u => 2,
        v => {} /* funky block
                 * comment */
                /* final comment */
    }
}

fn issue355() {
    match mac {
        a => println!("a", b),
        b => vec![1, 2],
        c => vec![3; 4],
        d => {
            println!("a", b)
        }
        e => {
            vec![1, 2]
        }
        f => {
            vec![3; 4]
        }
        h => println!("a", b), // h comment
        i => vec![1, 2],       // i comment
        j => vec![3; 4],       // j comment
        // k comment
        k => println!("a", b),
        // l comment
        l => vec![1, 2],
        // m comment
        m => vec![3; 4],
        // Rewrite splits macro
        nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn => {
            println!("a", b)
        }
        // Rewrite splits macro
        oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo => {
            vec![1, 2]
        }
        // Macro support fails to recognise this macro as splittable
        // We push the whole expr to a new line, TODO split this macro as well
        pppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppp => {
            vec![3; 4]
        }
        // q, r and s: Rewrite splits match arm
        qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq => {
            println!("a", b)
        }
        rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr => {
            vec![1, 2]
        }
        ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss => {
            vec![3; 4]
        }
        // Funky bracketing styles
        t => println! {"a", b},
        u => vec![1, 2],
        v => vec![3; 4],
        w => println!["a", b],
        x => vec![1, 2],
        y => vec![3; 4],
        // Brackets with comments
        tc => println! {"a", b}, // comment
        uc => vec![1, 2],        // comment
        vc => vec![3; 4],        // comment
        wc => println!["a", b],  // comment
        xc => vec![1, 2],        // comment
        yc => vec![3; 4],        // comment
        yd => looooooooooooooooooooooooooooooooooooooooooooooooooooooooong_func(
            aaaaaaaaaa, bbbbbbbbbb, cccccccccc, dddddddddd,
        ),
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
        LastImport { .. } => false,
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
            HTMLElementTypeId::HTMLCanvasElement,
        ))) => true,
        Some(NodeTypeId::Element(ElementTypeId::HTMLElement(
            HTMLElementTypeId::HTMLObjectElement,
        ))) => s.has_object_data(),
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

fn issue386() {
    match foo {
        BiEq | BiLt | BiLe | BiNe | BiGt | BiGe => true,
        BiAnd | BiOr | BiAdd | BiSub | BiMul | BiDiv | BiRem | BiBitXor | BiBitAnd | BiBitOr
        | BiShl | BiShr => false,
    }
}

fn guards() {
    match foo {
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
            if foooooooooooooo && barrrrrrrrrrrr => {}
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
        | aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
            if foooooooooooooo && barrrrrrrrrrrr => {}
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
            if fooooooooooooooooooooo
                && (bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
                    || cccccccccccccccccccccccccccccccccccccccc) => {}
    }
}

fn issue1371() {
    Some(match type_ {
        sfEvtClosed => Closed,
        sfEvtResized => {
            let e = unsafe { *event.size.as_ref() };

            Resized {
                width: e.width,
                height: e.height,
            }
        }
        sfEvtLostFocus => LostFocus,
        sfEvtGainedFocus => GainedFocus,
        sfEvtTextEntered => TextEntered {
            unicode: unsafe {
                ::std::char::from_u32((*event.text.as_ref()).unicode)
                    .expect("Invalid unicode encountered on TextEntered event")
            },
        },
        sfEvtKeyPressed => {
            let e = unsafe { event.key.as_ref() };

            KeyPressed {
                code: unsafe { ::std::mem::transmute(e.code) },
                alt: e.alt.to_bool(),
                ctrl: e.control.to_bool(),
                shift: e.shift.to_bool(),
                system: e.system.to_bool(),
            }
        }
        sfEvtKeyReleased => {
            let e = unsafe { event.key.as_ref() };

            KeyReleased {
                code: unsafe { ::std::mem::transmute(e.code) },
                alt: e.alt.to_bool(),
                ctrl: e.control.to_bool(),
                shift: e.shift.to_bool(),
                system: e.system.to_bool(),
            }
        }
    })
}

fn issue1395() {
    let bar = Some(true);
    let foo = Some(true);
    let mut x = false;
    bar.and_then(|_| match foo {
        None => None,
        Some(b) => {
            x = true;
            Some(b)
        }
    });
}

fn issue1456() {
    Ok(Recording {
        artists: match reader.evaluate(".//mb:recording/mb:artist-credit/mb:name-credit")? {
            Nodeset(nodeset) => {
                let res: Result<Vec<ArtistRef>, ReadError> = nodeset
                    .iter()
                    .map(|node| {
                        XPathNodeReader::new(node, &context).and_then(|r| ArtistRef::from_xml(&r))
                    })
                    .collect();
                res?
            }
            _ => Vec::new(),
        },
    })
}

fn issue1460() {
    let _ = match foo {
        REORDER_BUFFER_CHANGE_INTERNAL_SPEC_INSERT => {
            "internal_spec_insert_internal_spec_insert_internal_spec_insert"
        }
        _ => "reorder_something",
    };
}

fn issue525() {
    foobar(
        f,
        "{}",
        match *self {
            TaskState::Started => "started",
            TaskState::Success => "success",
            TaskState::Failed => "failed",
        },
    );
}

// #1838, #1839
fn match_with_near_max_width() {
    let (this_line_uses_99_characters_and_is_formatted_properly, x012345) = match some_expression {
        _ => unimplemented!(),
    };

    let (should_be_formatted_like_the_line_above_using_100_characters, x0) = match some_expression {
        _ => unimplemented!(),
    };

    let (should_put_the_brace_on_the_next_line_using_101_characters, x0000) = match some_expression
    {
        _ => unimplemented!(),
    };
    match m {
        Variant::Tag
        | Variant::Tag2
        | Variant::Tag3
        | Variant::Tag4
        | Variant::Tag5
        | Variant::Tag6 => {}
    }
}

fn match_with_trailing_spaces() {
    match x {
        Some(..) => 0,
        None => 1,
    }
}

fn issue_2099() {
    let a = match x {};
    let b = match x {};

    match x {}
}

// #2021
impl<'tcx> Const<'tcx> {
    pub fn from_constval<'a>() -> Const<'tcx> {
        let val = match *cv {
            ConstVal::Variant(_) | ConstVal::Aggregate(..) | ConstVal::Unevaluated(..) => bug!(
                "MIR must not use `{:?}` (aggregates are expanded to MIR rvalues)",
                cv
            ),
        };
    }
}

// #2151
fn issue_2151() {
    match either {
        x => {}
        y => (),
    }
}

// #2152
fn issue_2152() {
    match m {
        "aaaaaaaaaaaaa" | "bbbbbbbbbbbbb" | "cccccccccccccccccccccccccccccccccccccccccccc"
            if true => {}
        "bind" | "writev" | "readv" | "sendmsg" | "recvmsg" if android && (aarch64 || x86_64) => {
            true
        }
    }
}

// #2376
// Preserve block around expressions with condition.
fn issue_2376() {
    let mut x = None;
    match x {
        Some(0) => {
            for i in 1..11 {
                x = Some(i);
            }
        }
        Some(ref mut y) => {
            while *y < 10 {
                *y += 1;
            }
        }
        None => {
            while let None = x {
                x = Some(10);
            }
        }
    }
}

// #2621
// Strip leading `|` in match arm patterns
fn issue_2621() {
    let x = Foo::A;
    match x {
        Foo::A => println!("No vert single condition"),
        Foo::B | Foo::C => println!("Center vert two conditions"),
        Foo::D => println!("Preceding vert single condition"),
        Foo::E | Foo::F => println!("Preceding vert over two lines"),
        Foo::G | Foo::H => println!("Trailing vert over two lines"),
        // Comment on its own line
        Foo::I => println!("With comment"), // Comment after line
    }
}

fn issue_2377() {
    match tok {
        Tok::Not
        | Tok::BNot
        | Tok::Plus
        | Tok::Minus
        | Tok::PlusPlus
        | Tok::MinusMinus
        | Tok::Void
        | Tok::Delete
            if prec <= 16 =>
        {
            // code here...
        }
        Tok::TypeOf if prec <= 16 => {}
    }
}

// #3040
fn issue_3040() {
    {
        match foo {
            DevtoolScriptControlMsg::WantsLiveNotifications(id, to_send) => {
                match documents.find_window(id) {
                    Some(window) => {
                        devtools::handle_wants_live_notifications(window.upcast(), to_send)
                    }
                    None => return warn!("Message sent to closed pipeline {}.", id),
                }
            }
        }
    }
}

// #3030
fn issue_3030() {
    match input.trim().parse::<f64>() {
        Ok(val)
            if !(
                // A valid number is the same as what rust considers to be valid,
                // except for +1., NaN, and Infinity.
                val.is_infinite() || val.is_nan() || input.ends_with(".") || input.starts_with("+")
            ) => {}
    }
}

fn issue_3005() {
    match *token {
        Token::Dimension {
            value, ref unit, ..
        } if num_context.is_ok(context.parsing_mode, value) => {
            return NoCalcLength::parse_dimension(context, value, unit)
                .map(LengthOrPercentage::Length)
                .map_err(|()| location.new_unexpected_token_error(token.clone()));
        }
    }
}

// #3774
fn issue_3774() {
    {
        {
            {
                match foo {
                    Lam(_, _, _) | Pi(_, _, _) | Let(_, _, _, _) | Embed(_) | Var(_) => unreachab(),
                    Lam(_, _, _) | Pi(_, _, _) | Let(_, _, _, _) | Embed(_) | Var(_) => unreacha!(),
                    Lam(_, _, _) | Pi(_, _, _) | Let(_, _, _, _) | Embed(_) | Var(_) => {
                        unreachabl()
                    }
                    Lam(_, _, _) | Pi(_, _, _) | Let(_, _, _, _) | Embed(_) | Var(_) => {
                        unreachae!()
                    }
                    Lam(_, _, _) | Pi(_, _, _) | Let(_, _, _, _) | Embed(_) | Var(_) => {
                        unreachable()
                    }
                    Lam(_, _, _) | Pi(_, _, _) | Let(_, _, _, _) | Embed(_) | Var(_) => {
                        unreachable!()
                    }
                    Lam(_, _, _) | Pi(_, _, _) | Let(_, _, _, _) | Embed(_) | Var(_) => {
                        rrunreachable!()
                    }
                }
            }
        }
    }
}

// #4109
fn issue_4109() {
    match () {
        _ => {
            #[cfg(debug_assertions)]
            {
                println!("Foo");
            }
        }
    }

    match () {
        _ => {
            #[allow(unsafe_code)]
            unsafe {}
        }
    }
}
