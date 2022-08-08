#![warn(clippy::collapsible_str_replace)]

fn get_filter() -> &'static str {
    "u"
}

fn main() {
    let misspelled = "hesuo worpd";

    let p = 'p';
    let s = 's';
    let u = 'u';
    let l = "l";

    // LINT CASES
    let replacement = misspelled.replace('s', "l").replace('u', "l");
    println!("{replacement}");

    let replacement = misspelled.replace('s', l).replace('u', l);
    println!("{replacement}");

    let replacement = misspelled.replace('s', "l").replace('u', "l").replace('p', "l");
    println!("{replacement}");

    let replacement = misspelled
        .replace('s', "l")
        .replace('u', "l")
        .replace('p', "l")
        .replace('d', "l");
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

    // NO LINT CASES
    let replacement = misspelled.replace('s', "l");
    println!("{replacement}");

    let replacement = misspelled.replace(s, "l");
    println!("{replacement}");

    // If the consecutive `str::replace` calls have different `to` arguments, do not lint
    let replacement = misspelled.replace('s', "l").replace('u', "p");
    println!("{replacement}");

    let replacement = misspelled.replace(&get_filter(), "l");
    println!("{replacement}");

    let replacement = misspelled.replace(&['s', 'u', 'p'], "l");
    println!("{replacement}");

    let replacement = misspelled.replace(&['s', 'u', 'p'], l);
    println!("{replacement}");

    let replacement = misspelled.replace(&['s', 'u'], "l").replace(&['u', 'p'], "l");
    println!("{replacement}");

    let replacement = misspelled.replace('s', "l").replace(&['u', 'p'], "l");
    println!("{replacement}");

    let replacement = misspelled.replace(&['s', 'u'], "l").replace('p', "l");
    println!("replacement");

    let replacement = misspelled.replace(&['s', u, 'p'], "l");
    println!("{replacement}");

    let replacement = misspelled.replace(&[s, u, 'p'], "l");
    println!("{replacement}");

    let replacement = misspelled.replace(&[s, u, p], "l");
    println!("{replacement}");

    let replacement = misspelled.replace(&[s, u], "l").replace(&[u, p], "l");
    println!("{replacement}");
}
