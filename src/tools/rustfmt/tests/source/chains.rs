// rustfmt-use_small_heuristics: Off
// Test chain formatting.

fn main() {
    let a = b .c
    .d.1
                .foo(|x| x + 1);

    bbbbbbbbbbbbbbbbbbb.ccccccccccccccccccccccccccccccccccccc
                       .ddddddddddddddddddddddddddd();

    bbbbbbbbbbbbbbbbbbb.ccccccccccccccccccccccccccccccccccccc.ddddddddddddddddddddddddddd.eeeeeeee();

    let f = fooooooooooooooooooooooooooooooooooooooooooooooooooo.baaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaar;

    // Test case where first chain element isn't a path, but is shorter than
    // the size of a tab.
    x()
        .y(|| match cond() { true => (), false => () });

    loong_func()
        .quux(move || if true {
            1
        } else {
            2
        });

    some_fuuuuuuuuunction()
        .method_call_a(aaaaa, bbbbb, |c| {
            let x = c;
            x
        });

    some_fuuuuuuuuunction().method_call_a(aaaaa, bbbbb, |c| {
        let x = c;
        x
    }).method_call_b(aaaaa, bbbbb, |c| {
        let x = c;
        x
    });

    fffffffffffffffffffffffffffffffffff(a,
                                        {
                                            SCRIPT_TASK_ROOT
                                            .with(|root| {
                                                *root.borrow_mut()  =   Some(&script_task);
                                            });
                                        });                                        

    let suuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuum = xxxxxxx
        .map(|x| x + 5)
        .map(|x| x / 2)
        .fold(0, |acc, x| acc + x);

    body.fold(Body::new(), |mut body, chunk| {
        body.extend(chunk);
        Ok(body)
    }).and_then(move |body| {
            let req = Request::from_parts(parts, body);
            f(req).map_err(|_| io::Error::new(io::ErrorKind::Other, ""))
        });

    aaaaaaaaaaaaaaaa.map(|x| {
                         x += 1;
                         x
                     }).filter(some_mod::some_filter)
}

fn floaters() {
    let z = Foo {
        field1: val1,
        field2: val2,
    };

    let x = Foo {
        field1: val1,
        field2: val2,
    }.method_call().method_call();

    let y = if cond {
                val1
            } else {
                val2
            }
                .method_call();

    {
        match x {
            PushParam => {
                // params are 1-indexed
                stack.push(mparams[match cur.to_digit(10) {
                    Some(d) => d as usize - 1,
                    None => return Err("bad param number".to_owned()),
                }]
                               .clone());
            }
        }
    }

    if cond { some(); } else { none(); }
        .bar()
        .baz();

    Foo { x: val } .baz(|| { force(); multiline();    }) .quux(); 

    Foo { y: i_am_multi_line, z: ok }
        .baz(|| {
            force(); multiline();
        })
        .quux(); 

    a + match x { true => "yay!", false => "boo!" }.bar()
}

fn is_replaced_content() -> bool {
    constellat.send(ConstellationMsg::ViewportConstrained(
            self.id, constraints)).unwrap();
}

fn issue587() {
    a.b::<()>(c);

    std::mem::transmute(dl.symbol::<()>("init").unwrap())
}

fn try_shorthand() {
    let x = expr?;
    let y = expr.kaas()?.test();
    let loooooooooooooooooooooooooooooooooooooooooong = does_this?.look?.good?.should_we_break?.after_the_first_question_mark?;
    let yyyy = expr?.another?.another?.another?.another?.another?.another?.another?.another?.test();
    let zzzz = expr?.another?.another?.another?.another?;
    let aaa =  x ????????????  ?????????????? ???? ?????  ?????????????? ?????????  ?????????????? ??;

    let y = a.very .loooooooooooooooooooooooooooooooooooooong()  .chain()
             .inside()          .weeeeeeeeeeeeeee()? .test()  .0
        .x;

                parameterized(f,
                              substs,
                              def_id,
                              Ns::Value,
                              &[],
                              |tcx| tcx.lookup_item_type(def_id).generics)?;
    fooooooooooooooooooooooooooo()?.bar()?.baaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaz()?;
}

fn issue_1004() {
         match *self {
                ty::ImplOrTraitItem::MethodTraitItem(ref i) => write!(f, "{:?}", i),
                ty::ImplOrTraitItem::ConstTraitItem(ref i) => write!(f, "{:?}", i),
                ty::ImplOrTraitItem::TypeTraitItem(ref i) => write!(f, "{:?}", i),
            }
            ?;

            ty::tls::with(|tcx| {
                let tap = ty::Binder(TraitAndProjections(principal, projections));
                in_binder(f, tcx, &ty::Binder(""), Some(tap))
            })
            ?;
}

fn issue1392() {
    test_method(r#"
        if foo {
            a();
        }
        else {
            b();
        }
        "#.trim());
}

// #2067
impl Settings {
    fn save(&self) -> Result<()> {
        let mut file = File::create(&settings_path).chain_err(|| ErrorKind::WriteError(settings_path.clone()))?;
    }
}

fn issue2126() {
    {
        {
            {
                {
                    {
                        let x = self.span_from(sub_span.expect("No span found for struct arant variant"));
                        self.sspanpan_from_span(sub_span.expect("No span found for struct variant"));
                        let x = self.spanpan_from_span(sub_span.expect("No span found for struct variant"))?;
                    }
                }
            }
        }
    }
}

// #2200
impl Foo {
    pub fn from_ast(diagnostic: &::errors::Handler,
                    attrs: &[ast::Attribute]) -> Attributes {
        let other_attrs = attrs.iter().filter_map(|attr| {
            attr.with_desugared_doc(|attr| {
                if attr.check_name("doc") {
                    if let Some(mi) = attr.meta() {
                        if let Some(value) = mi.value_str() {
                            doc_strings.push(DocFragment::Include(line,
                                                                  attr.span,
                                                                  filename,
                                                                  contents));
                        }
                    }
                }
            })
        }).collect();
    }
}

// #2415
// Avoid orphan in chain
fn issue2415() {
    let base_url = (|| {
        // stuff

        Ok((|| {
            // stuff
            Some(value.to_string())
        })()
           .ok_or("")?)
    })()
        .unwrap_or_else(|_: Box<::std::error::Error>| String::from(""));
}

impl issue_2786 {
    fn thing(&self) {
        foo(|a| {
            println!("a");
            println!("b");
        }).bar(|c| {
            println!("a");
            println!("b");
        })
            .baz(|c| {
                println!("a");
                println!("b");
            })
    }
}

fn issue_2773() {
    let bar = Some(0);
    bar.or_else(|| {
        // do stuff
        None
    }).or_else(|| {
            // do other stuff
            None
        })
        .and_then(|val| {
            // do this stuff
            None
        });
}

fn issue_3034() {
    disallowed_headers.iter().any(|header| *header == name) ||
        disallowed_header_prefixes.iter().any(|prefix| name.starts_with(prefix))
}
