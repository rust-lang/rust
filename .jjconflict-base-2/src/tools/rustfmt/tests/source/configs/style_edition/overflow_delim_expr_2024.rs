// rustfmt-style_edition: 2024

fn combine_blocklike() {
    do_thing(
        |param| {
            action();
            foo(param)
        },
    );

    do_thing(
        x,
        |param| {
            action();
            foo(param)
        },
    );

    do_thing(
        x,

        // I'll be discussing the `action` with your para(m)legal counsel
        |param| {
            action();
            foo(param)
        },
    );

    do_thing(
        Bar {
            x: value,
            y: value2,
        },
    );

    do_thing(
        x,
        Bar {
            x: value,
            y: value2,
        },
    );

    do_thing(
        x,

        // Let me tell you about that one time at the `Bar`
        Bar {
            x: value,
            y: value2,
        },
    );

    do_thing(
        &[
            value_with_longer_name,
            value2_with_longer_name,
            value3_with_longer_name,
            value4_with_longer_name,
        ],
    );

    do_thing(
        x,
        &[
            value_with_longer_name,
            value2_with_longer_name,
            value3_with_longer_name,
            value4_with_longer_name,
        ],
    );

    do_thing(
        x,

        // Just admit it; my list is longer than can be folded on to one line
        &[
            value_with_longer_name,
            value2_with_longer_name,
            value3_with_longer_name,
            value4_with_longer_name,
        ],
    );

    do_thing(
        vec![
            value_with_longer_name,
            value2_with_longer_name,
            value3_with_longer_name,
            value4_with_longer_name,
        ],
    );

    do_thing(
        x,
        vec![
            value_with_longer_name,
            value2_with_longer_name,
            value3_with_longer_name,
            value4_with_longer_name,
        ],
    );

    do_thing(
        x,

        // Just admit it; my list is longer than can be folded on to one line
        vec![
            value_with_longer_name,
            value2_with_longer_name,
            value3_with_longer_name,
            value4_with_longer_name,
        ],
    );

    do_thing(
        x,
        (
            1,
            2,
            3,
            |param| {
                action();
                foo(param)
            },
        ),
    );
}

fn combine_struct_sample() {
    let identity = verify(
        &ctx,
        VerifyLogin {
            type_: LoginType::Username,
            username: args.username.clone(),
            password: Some(args.password.clone()),
            domain: None,
        },
    )?;
}

fn combine_macro_sample() {
    rocket::ignite()
        .mount(
            "/",
            routes![
                http::auth::login,
                http::auth::logout,
                http::cors::options,
                http::action::dance,
                http::action::sleep,
            ],
        )
        .launch();
}
