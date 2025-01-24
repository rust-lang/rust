// ignore-tidy-tab

fn main() {
    println!("{");
    //~^ ERROR invalid format string: expected `}` but string was terminated
    println!("{{}}");
    println!("}");
    //~^ ERROR invalid format string: unmatched `}` found
    let _ = format!("{_}", _ = 6usize);
    //~^ ERROR invalid format string: invalid argument name `_`
    let _ = format!("{a:_}", a = "", _ = 0);
    //~^ ERROR invalid format string: invalid argument name `_`
    let _ = format!("{a:._$}", a = "", _ = 0);
    //~^ ERROR invalid format string: invalid argument name `_`
    let _ = format!("{");
    //~^ ERROR invalid format string: expected `}` but string was terminated
    let _ = format!("}");
    //~^ ERROR invalid format string: unmatched `}` found
    let _ = format!("{\\}");
    //~^ ERROR invalid format string: expected `}`, found `\`
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
    println!("{} {} {}", 1, 2);
    //~^ ERROR 3 positional arguments in format string, but there are 2 arguments
}
