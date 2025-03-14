#![allow(unused)]
#![warn(clippy::collapsible_str_replace)]

fn get_filter() -> char {
    'u'
}

fn main() {
    let d = 'd';
    let p = 'p';
    let s = 's';
    let u = 'u';
    let l = "l";

    let mut iter = ["l", "z"].iter();

    // LINT CASES
    let _ = "hesuo worpd".replace('s', "l").replace('u', "l");
    //~^ collapsible_str_replace

    let _ = "hesuo worpd".replace('s', l).replace('u', l);
    //~^ collapsible_str_replace

    let _ = "hesuo worpd".replace('s', "l").replace('u', "l").replace('p', "l");
    //~^ collapsible_str_replace

    let _ = "hesuo worpd"
        .replace('s', "l")
        //~^ collapsible_str_replace
        .replace('u', "l")
        .replace('p', "l")
        .replace('d', "l");

    let _ = "hesuo world".replace(s, "l").replace('u', "l");
    //~^ collapsible_str_replace

    let _ = "hesuo worpd".replace(s, "l").replace('u', "l").replace('p', "l");
    //~^ collapsible_str_replace

    let _ = "hesuo worpd".replace(s, "l").replace(u, "l").replace('p', "l");
    //~^ collapsible_str_replace

    let _ = "hesuo worpd".replace(s, "l").replace(u, "l").replace(p, "l");
    //~^ collapsible_str_replace

    let _ = "hesuo worlp".replace('s', "l").replace('u', "l").replace('p', "d");
    //~^ collapsible_str_replace

    let _ = "hesuo worpd".replace('s', "x").replace('u', "l").replace('p', "l");
    //~^ collapsible_str_replace

    // Note: Future iterations could lint `replace(|c| matches!(c, "su" | 'd' | 'p'), "l")`
    let _ = "hesudo worpd".replace("su", "l").replace('d', "l").replace('p', "l");
    //~^ collapsible_str_replace

    let _ = "hesudo worpd".replace(d, "l").replace('p', "l").replace("su", "l");
    //~^ collapsible_str_replace

    let _ = "hesuo world".replace(get_filter(), "l").replace('s', "l");
    //~^ collapsible_str_replace

    // NO LINT CASES
    let _ = "hesuo world".replace('s', "l").replace('u', "p");

    let _ = "hesuo worpd".replace('s', "l").replace('p', l);

    let _ = "hesudo worpd".replace('d', "l").replace("su", "l").replace('p', "l");

    // Note: Future iterations of `collapsible_str_replace` might lint this and combine to `[s, u, p]`
    let _ = "hesuo worpd".replace([s, u], "l").replace([u, p], "l");

    let _ = "hesuo worpd".replace(['s', 'u'], "l").replace(['u', 'p'], "l");

    let _ = "hesuo worpd".replace('s', "l").replace(['u', 'p'], "l");

    let _ = "hesuo worpd".replace(['s', 'u', 'p'], "l").replace('r', "l");

    let _ = "hesuo worpd".replace(['s', 'u', 'p'], l).replace('r', l);

    let _ = "hesuo worpd".replace(['s', u, 'p'], "l").replace('r', "l");

    let _ = "hesuo worpd".replace([s, u], "l").replace(p, "l");

    // Regression test
    let _ = "hesuo worpd"
        .replace('u', iter.next().unwrap())
        .replace('s', iter.next().unwrap());
}

#[clippy::msrv = "1.57"]
fn msrv_1_57() {
    let _ = "".replace('a', "1.57").replace('b', "1.57");
}

#[clippy::msrv = "1.58"]
fn msrv_1_58() {
    let _ = "".replace('a', "1.58").replace('b', "1.58");
    //~^ collapsible_str_replace
}
