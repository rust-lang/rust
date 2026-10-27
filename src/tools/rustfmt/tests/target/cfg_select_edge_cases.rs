//@ rustfmt-unstable: true
//@ rustfmt-edition: 2024

// We need to make sure we're not adding or removing important braces

// <https://github.com/rust-lang/rustfmt/issues/7045#issuecomment-6182909709>
struct LoudDrop(&'static str);
impl Drop for LoudDrop {
    fn drop(&mut self) {
        println!("drop {}", self.0);
    }
}

fn main() {
    cfg_select! {
        _ => {{ (LoudDrop("temp"), LoudDrop("value")).1 }}
    };
    println!("end");
}

// <https://github.com/rust-lang/rustfmt/issues/7045#issuecomment-6183194180>
fn works() -> i32 {
    {
        1
    }
    cfg_select! {
        _ => {}
    }
}

fn fails() -> i32 {
    {
        1
    }
    cfg_select! {
        _ => {{}}
    }
}

// <https://github.com/rust-lang/rustfmt/issues/7045#issuecomment-6183219250>
fn works() {
    let x = cfg_select! {
        _ =>
        {
            #[cfg(true)]
            1
        }
    };
}

fn fails() {
    let x = cfg_select! {
        // rustfmt will add braces here
        _ =>
        {
            #[cfg(true)]
            1
        }
    };
}

// Note, the weird brace placement above seems to be an existing issue with match arms.
fn check_match_block_formatting() {
    match x {
        _ =>
        {
            #[cfg(true)]
            1
        }
    }
}

fn cfg_select_comments_that_might_get_lost() {
    cfg_select! {
        _ => {
            // after opening brace
            { x }
            // before closing brace
        }
    }
    cfg_select! {
        _ => {
            /* after opening brace */
            { x }
            /* before closing brace */
        }
    }
    cfg_select! {
        _ => {{ x }}
    }
    cfg_select! {
        _ => {{ x }}
    }

    cfg_select! {
        three_blocks_exclamation_point => {{
            /* a */
            {
                b;
                c
            } /* d */
        }},
        // No trailing comma in source
        three_blocks_exclamation_point => {{
            /* a */
            {
                b;
                c
            } /* d */
        }}
    }
}

fn other_cfg_select_cases() {
    cfg_select! {
        three_blocks_exclamation_point => {{
            {
                a;
                b
            }
        }},
        // No trailing comma in source
        three_blocks_exclamation_point => {{
            {
                a;
                b
            }
        }}
    }

    cfg_select! {
        outer_unsafe_block => unsafe { { .. } },
        // No trailing comma in source
        outer_unsafe_block => unsafe { { .. } },
    }
    cfg_select! {
        outer_const_block => const { { .. } },
        // No trailing comma in source
        outer_const_block => const { { .. } },
    }
    cfg_select! {
        outer_async_block => async { { .. } },
        // No trailing comma in source
        outer_async_block => async { { .. } },
    }
    cfg_select! {
        outer_gen_block => gen { { .. } },
        // No trailing comma in source
        outer_gen_block => gen { { .. } },
    }
    cfg_select! {
        outer_try_block => try { { .. } },
        // No trailing comma in source
        outer_try_block => try { { .. } },
    }
    cfg_select! {
        outer_block_with_label => 'a: {
            { .. }
        }
        // No trailing comma in source
        outer_block_with_label => 'a: {
            { .. }
        }
    }

    // outer attr syntax currently isn't valid syntax.
    // Maybe that will change in the future so keeping this test case around.
    cfg_select! {
        outer_block_with_outer_attr =>
        #[foo]
        {
            { .. }
        }
        // No trailing comma in source
        outer_block_with_outer_attr =>
        #[foo]
        {
            { .. }
        }
    }
    cfg_select! {
        outer_block_with_inner_attr =>
        {
            #![foo]
            { .. }
        }
        // No trailing comma in source
        outer_block_with_inner_attr =>
        {
            #![foo]
            { .. }
        }
    }

    cfg_select! {
        inner_unsafe_block => {unsafe { .. }},
        // No trailing comma in source
        inner_unsafe_block => {unsafe { .. }}
    }
    cfg_select! {
        inner_const_block => const { .. },
        // No trailing comma in source
        inner_const_block => const { .. },
    }
    cfg_select! {
        inner_async_block => async { .. },
        // No trailing comma in source
        inner_async_block => async { .. },
    }
    cfg_select! {
        inner_gen_block => gen { .. },
        // No trailing comma in source
        inner_gen_block => gen { .. },
    }
    cfg_select! {
        inner_try_block => try { .. },
        // No trailing comma in source
        inner_try_block => try { .. },
    }
    cfg_select! {
        inner_block_with_label => {'a: { .. }},
        // No trailing comma in source
        inner_block_with_label => {'a: { .. }}
    }
    cfg_select! {
        inner_block_with_outer_attr => {#[foo]
        {
            ..
        }},
        // No trailing comma in source
        inner_block_with_outer_attr => {#[foo]
        {
            ..
        }}
    }
    cfg_select! {
        inner_block_with_inner_attr => {{
            #![foo]
        }},
        // No trailing comma in source
        inner_block_with_inner_attr => {{
            #![foo]
        }}
    }

    cfg_select! {
        _ => v,
    }
    .push(2); // place expr

    // No trailing comma in source
    cfg_select! {
        _ => v,
    }
    .push(2); // place expr

    cfg_select! {
        _ => {{ v }},
    }
    .push(3); // value expr

    // No trailing comma in source
    cfg_select! {
        _ => {{ v }}
    }
    .push(3); // value expr

    // Doc comments
    cfg_select! {
        // rustfmt adds braces
        _ => {
            (///
            {})
        }
    }

    cfg_select! {
        _ => {{
            //!
            {}
        }}
    }

    cfg_select! {
        _ =>
        {
            #[inline]
            || {}
        }
    };
}

fn long_predicate() {
    let a = cfg_select! {
        all(
            target_os = "linux",
            target_arch = "x86_64",
            target_env = "gnu",
            feature = "hello",
            feature = "world"
        ) => {{ x }},
    };
}

fn with_trailing_comma_fits_max_width() {
    cfg_select! {
        // max_width = 100 ------------------------------------------------------------------------|
        all(target_os = "linux", target_arch = "x86_64", target_env = "gnu") => {{ xxxxxxxxxxxxx }},
        all(
            target_os = "linux",
            target_arch = "x86_64",
            target_env = "gnu",
            feature = "hello",
            feature = "world"
        ) => {{ xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx }},
        _ => {{ xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx }},
    }
}

fn with_trailing_comma_exceeds_max_width() {
    cfg_select! {
        // max_width = 100 ------------------------------------------------------------------------|
        all(target_os = "linux", target_arch = "x86_64", target_env = "gnu") => {{
            xxxxxxxxxxxxxx
        }},
        all(
            target_os = "linux",
            target_arch = "x86_64",
            target_env = "gnu",
            feature = "hello",
            feature = "world"
        ) => {{
            xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        }},
        _ => {{
            xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        }},
    }
}

fn without_trailing_comma_fits_max_width() {
    cfg_select! {
        // max_width = 100 ------------------------------------------------------------------------|
        all(target_os = "linux", target_arch = "x86_64", target_env = "gnu") => {{ xxxxxxxxxxxxxx }}
        all(
            target_os = "linux",
            target_arch = "x86_64",
            target_env = "gnu",
            feature = "hello",
            feature = "world"
        ) => {{ xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx }}
        _ => {{ xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx }}
    }
}

fn without_trailing_comma_exceeds_max_width() {
    cfg_select! {
        // max_width = 100 ------------------------------------------------------------------------|
        all(target_os = "linux", target_arch = "x86_64", target_env = "gnu") => {{
            xxxxxxxxxxxxxxx
        }}
        all(
            target_os = "linux",
            target_arch = "x86_64",
            target_env = "gnu",
            feature = "hello",
            feature = "world"
        ) => {{
            xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        }}
        _ => {{
            xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        }}
    }
}
