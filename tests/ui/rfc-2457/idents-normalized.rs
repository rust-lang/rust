// check-pass

struct Résumé; // ['LATIN SMALL LETTER E WITH ACUTE']

fn main() {
    let _ = Résumé; // ['LATIN SMALL LETTER E', 'COMBINING ACUTE ACCENT']
}
