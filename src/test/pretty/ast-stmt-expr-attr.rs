// pp-exact

fn main() {}

#[cfg(FALSE)]
fn syntax() {
    let _ = #[attr] box 0;
    let _ = #[attr] [];
    let _ = #[attr] [0];
    let _ = #[attr] [0; 0];
    let _ = #[attr] [0, 0, 0];
    let _ = #[attr] foo();
    let _ = #[attr] x.foo();
    let _ = #[attr] ();
    let _ = #[attr] (#[attr] 0,);
    let _ = #[attr] (#[attr] 0, 0);
    let _ = #[attr] 0 + #[attr] 0;
    let _ = #[attr] 0 / #[attr] 0;
    let _ = #[attr] 0 & #[attr] 0;
    let _ = #[attr] 0 % #[attr] 0;
    let _ = #[attr] (0 + 0);
    let _ = #[attr] !0;
    let _ = #[attr] -0;
    let _ = #[attr] false;
    let _ = #[attr] 0;
    let _ = #[attr] 'c';
    let _ = #[attr] x as Y;
    let _ = #[attr] (x as Y);
    let _ =
        #[attr] while true {
                    #![attr]
                };
    let _ =
        #[attr] while let Some(false) = true {
                    #![attr]
                };
    let _ =
        #[attr] for x in y {
                    #![attr]
                };
    let _ =
        #[attr] loop {
                    #![attr]
                };
    let _ =
        #[attr] match true {
                    #![attr]
                            #[attr]
                            _ => false,
                };
    let _ = #[attr] || #[attr] foo;
    let _ = #[attr] move || #[attr] foo;
    let _ =
        #[attr] ||
                    #[attr] {
                                #![attr]
                                foo
                            };
    let _ =
        #[attr] move ||
                    #[attr] {
                                #![attr]
                                foo
                            };
    let _ =
        #[attr] ||
                    {
                        #![attr]
                        foo
                    };
    let _ =
        #[attr] move ||
                    {
                        #![attr]
                        foo
                    };
    let _ =
        #[attr] {
                    #![attr]
                };
    let _ =
        #[attr] {
                    #![attr]
                    let _ = ();
                };
    let _ =
        #[attr] {
                    #![attr]
                    let _ = ();
                    foo
                };
    let _ = #[attr] x = y;
    let _ = #[attr] (x = y);
    let _ = #[attr] x += y;
    let _ = #[attr] (x += y);
    let _ = #[attr] foo.bar;
    let _ = (#[attr] foo).bar;
    let _ = #[attr] foo.0;
    let _ = (#[attr] foo).0;
    let _ = #[attr] foo[bar];
    let _ = (#[attr] foo)[bar];
    let _ = #[attr] 0..#[attr] 0;
    let _ = #[attr] 0..;
    let _ = #[attr] (0..0);
    let _ = #[attr] (0..);
    let _ = #[attr] (..0);
    let _ = #[attr] (..);
    let _ = #[attr] foo::bar::baz;
    let _ = #[attr] &0;
    let _ = #[attr] &mut 0;
    let _ = #[attr] &#[attr] 0;
    let _ = #[attr] &mut #[attr] 0;
    let _ = #[attr] break;
    let _ = #[attr] continue;
    let _ = #[attr] return;
    let _ = #[attr] foo!();
    let _ = #[attr] foo!(#! [attr]);
    let _ = #[attr] foo![];
    let _ = #[attr] foo![#! [attr]];
    let _ = #[attr] foo! {};
    let _ = #[attr] foo! { #! [attr] };
    let _ = #[attr] Foo{bar: baz,};
    let _ = #[attr] Foo{..foo};
    let _ = #[attr] Foo{bar: baz, ..foo};
    let _ = #[attr] (0);

    {
        #[attr]
        let _ = 0;

        #[attr]
        0;

        #[attr]
        foo!();

        #[attr]
        foo! {}

        #[attr]
        foo![];
    }

    {
        #[attr]
        let _ = 0;
    }
    {

        #[attr]
        0
    }
    {

        #[attr]
        {
            #![attr]
        }
    }
    {

        #[attr]
        foo!()
    }
    {

        #[attr]
        foo![]
    }
    {

        #[attr]
        foo! {}
    }
}
