#![warn(clippy::collapsible_str_replace)]

fn get_filter() -> &'static str {
    "u"
}

fn main() {
    let misspelled = "hesuo worpd";

    let p = 'p';
    let s = 's';
    let u = 'u';

    // LINT CASES
    // If the first argument to a single `str::replace` call is a slice and none of the chars
    // are variables, recommend `collapsible_str_replace`
    let replacement = misspelled.replace(&['s', 'u', 'p'], "l");
    println!("{replacement}");

    // If there are consecutive calls to `str::replace` and none of the chars are variables,
    // recommend `collapsible_str_replace`
    let replacement = misspelled.replace('s', "l").replace('u', "l");
    println!("{replacement}");

    let replacement = misspelled.replace('s', "l").replace('u', "l").replace('p', "l");
    println!("{replacement}");

    // NO LINT CASES
    // If there is a single call to `str::replace` and the first argument is a char or a variable,
    // do not recommend `collapsible_str_replace`
    let replacement = misspelled.replace('s', "l");
    println!("{replacement}");

    let replacement = misspelled.replace(s, "l");
    println!("{replacement}");

    // If the `from` argument is of kind other than a slice or a char, do not lint
    let replacement = misspelled.replace(&get_filter(), "l");

    // NO LINT TIL IMPROVEMENT
    // If multiple `str::replace` calls contain slices and none of the chars are variables,
    // the first iteration does not recommend `collapsible_str_replace`
    let replacement = misspelled.replace(&['s', 'u', 'p'], "l").replace(&['s', 'p'], "l");
    println!("{replacement}");

    // If a mixture of `str::replace` calls with slice and char arguments are used for `from` arg,
    // the first iteration does not recommend `collapsible_str_replace`
    let replacement = misspelled.replace(&['s', 'u'], "l").replace('p', "l");
    println!("replacement");

    let replacement = misspelled.replace('p', "l").replace(&['s', 'u'], "l");

    // The first iteration of `collapsible_str_replace` will not create lint if the first argument to
    // a single `str::replace` call is a slice and one or more of its chars are variables
    let replacement = misspelled.replace(&['s', u, 'p'], "l");
    println!("{replacement}");

    let replacement = misspelled.replace(&[s, u, 'p'], "l");
    println!("{replacement}");

    let replacement = misspelled.replace(&[s, u, p], "l");
    println!("{replacement}");

    let replacement = misspelled.replace(&[s, u], "l").replace(&[u, p], "l");
    println!("{replacement}");

    // FALLBACK CASES
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
