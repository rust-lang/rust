// rustfmt-indent_style: Visual
// Test chain formatting.

fn main() {
    // Don't put chains on a single line if it wasn't so in source.
    let a = b .c
    .d.1
                .foo(|x| x + 1);

    bbbbbbbbbbbbbbbbbbb.ccccccccccccccccccccccccccccccccccccc
                       .ddddddddddddddddddddddddddd();

    bbbbbbbbbbbbbbbbbbb.ccccccccccccccccccccccccccccccccccccc.ddddddddddddddddddddddddddd.eeeeeeee();

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

fn issue_1389() {
    let names = String::from_utf8(names)?.split('|').map(str::to_owned).collect();
}

fn issue1217() -> Result<Mnemonic, Error> {
let random_chars: String = OsRng::new()?
    .gen_ascii_chars()
    .take(self.bit_length)
    .collect();

    Ok(Mnemonic::new(&random_chars))
}

fn issue1236(options: Vec<String>) -> Result<Option<String>> {
let process = Command::new("dmenu").stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .spawn()
    .chain_err(|| "failed to spawn dmenu")?;
}

fn issue1434() {
    for _ in 0..100 {
        let prototype_id = PrototypeIdData::from_reader::<_, B>(&mut self.file_cursor).chain_err(|| {
            format!("could not read prototype ID at offset {:#010x}",
                    current_offset)
        })?;
    }
}

fn issue2264() {
    {
        something.function()
            .map(|| {
                if let a_very_very_very_very_very_very_very_very_long_variable =
                    compute_this_variable()
                {
                    println!("Hello");
                }
            })
            .collect();
    }
}
