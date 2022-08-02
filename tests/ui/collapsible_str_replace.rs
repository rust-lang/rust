#![warn(clippy::collapsible_str_replace)]

fn main() {
    let misspelled = "hesuo worpd";

    let p = 'p';
    let s = 's';
    let u = 'u';

    // If the first argument to a single `str::replace` call is a slice and none of the chars
    // are variables, recommend `collapsible_str_replace`
    let replacement = misspelled.replace(&['s', 'u', 'p'], "l");
    println!("{replacement}");

    // The first iteration of `collapsible_str_replace` will not create lint if the first argument to
    // a single `str::replace` call is a slice and one or more of its chars are variables
    let replacement = misspelled.replace(&['s', u, 'p'], "l");
    println!("{replacement}");

    let replacement = misspelled.replace(&[s, u, 'p'], "l");
    println!("{replacement}");

    let replacement = misspelled.replace(&[s, u, p], "l");
    println!("{replacement}");

    // If there is a single call to `str::replace` and the first argument is a char or a variable, don't
    // not recommend `collapsible_str_replace`
    let replacement = misspelled.replace('s', "l");
    println!("{replacement}");

    let replacement = misspelled.replace(s, "l");
    println!("{replacement}");

    // If there are consecutive calls to `str::replace` and none of the chars are variables,
    // recommend `collapsible_str_replace`
    let replacement = misspelled.replace('s', "l").replace('u', "l");
    println!("{replacement}");

    let replacement = misspelled.replace('s', "l").replace('u', "l").replace('p', "l");
    println!("{replacement}");

    // If there are consecutive calls to `str::replace` and all or any chars are variables,
    // recommend the fallback `misspelled.replace(&[s, u, p], "l")`
    let replacement = misspelled.replace(s, "l").replace('u', "l");
    println!("{replacement}");

    let replacement = misspelled.replace(s, "l").replace('u', "l").replace('p', "l");
    println!("{replacement}");

    let replacement = misspelled.replace(s, "l").replace(u, "l").replace('p', "l");
    println!("{replacement}");

    let replacement = misspelled.replace(s, "l").replace(u, "l").replace(p, "l");
    println!("{replacement}");
}
