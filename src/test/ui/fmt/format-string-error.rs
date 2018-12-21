// ignore-tidy-tab

fn main() {
    println!("{");
    //~^ ERROR invalid format string: expected `'}'` but string was terminated
    println!("{{}}");
    println!("}");
    //~^ ERROR invalid format string: unmatched `}` found
    let _ = format!("{_foo}", _foo = 6usize);
    //~^ ERROR invalid format string: invalid argument name `_foo`
    let _ = format!("{_}", _ = 6usize);
    //~^ ERROR invalid format string: invalid argument name `_`
    let _ = format!("{");
    //~^ ERROR invalid format string: expected `'}'` but string was terminated
    let _ = format!("}");
    //~^ ERROR invalid format string: unmatched `}` found
    let _ = format!("{\\}");
    //~^ ERROR invalid format string: expected `'}'`, found `'\\'`
    let _ = format!("\n\n\n{\n\n\n");
    //~^ ERROR invalid format string
    let _ = format!(r###"



	{"###);
    //~^ ERROR invalid format string
    let _ = format!(r###"



	{

"###);
    //~^ ERROR invalid format string
    let _ = format!(r###"



	}

"###);
    //~^^^ ERROR invalid format string
    let _ = format!(r###"



        }

"###);
    //~^^^ ERROR invalid format string: unmatched `}` found
}
