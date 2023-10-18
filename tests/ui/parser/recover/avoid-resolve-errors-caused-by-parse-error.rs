struct Website {
    url: String,
    title: Option<String>,
}

fn main() {
    let website = Website {
        url: "http://www.example.com".into(),
        title: Some("Example Domain".into()),
    };

    if let Website { url, Some(title) } = website { //~ ERROR expected `,`
        println!("[{}]({})", title, url);
    }
    if let Website { url, #, title } = website { //~ ERROR expected one of `!` or `[`, found `,`
        println!("[{}]({})", title, url);
    }
}
