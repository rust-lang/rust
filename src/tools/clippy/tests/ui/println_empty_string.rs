#![allow(clippy::match_single_binding)]

fn main() {
    println!();
    println!("");
    //~^ println_empty_string

    match "a" {
        _ => println!(""),
        //~^ println_empty_string
    }

    eprintln!();
    eprintln!("");
    //~^ println_empty_string

    match "a" {
        _ => eprintln!(""),
        //~^ println_empty_string
    }
}

#[rustfmt::skip]
fn issue_16167() {
    //~v println_empty_string
    println!(
        "\
            \
            "
            ,
    );

    match "a" {
        _ => println!("" ,), // there is a space between "" and comma
        //~^ println_empty_string
    }

    eprintln!(""	,); // there is a tab between "" and comma
    //~^ println_empty_string

    match "a" {
        _ => eprintln!(""	 ,), // tab and space between "" and comma
        //~^ println_empty_string
    }
}
