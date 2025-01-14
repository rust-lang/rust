struct Website {
    url: String,
    title: Option<String>,
}

enum Foo {
    Bar { a: i32 },
}

fn main() {
    let website = Website {
        url: "http://www.example.com".into(),
        title: Some("Example Domain".into()),
    };

    if let Website { url, Some(title) } = website { //~ ERROR expected `,`
        //~^ NOTE while parsing the fields for this pattern
        println!("[{}]({})", title, url); // we hide the errors for `title` and `url`
    }

    if let Website { url, .. } = website { //~ NOTE this pattern
        println!("[{}]({})", title, url); //~ ERROR cannot find value `title` in this scope
        //~^ NOTE not found in this scope
    }

    let x = Foo::Bar { a: 1 };
    if let Foo::Bar { .. } = x { //~ NOTE this pattern
        println!("{a}"); //~ ERROR cannot find value `a` in this scope
    }
}
