//@ check-pass
// Verify that sound fn pointer casts are NOT rejected by the V1 fix.

// Sound: no lifetime parameters at all
fn no_lifetimes(x: i32) -> i32 { x }
fn test_no_lifetimes() {
    let _f: fn(i32) -> i32 = no_lifetimes;
}

// Sound: implied bounds preserved in target
fn preserved<'a, 'b: 'a>(x: &'a &'b (), v: &'b i32) -> &'a i32 { v }
fn test_preserved() {
    let _f: fn(&'static &'static (), &'static i32) -> &'static i32 = preserved;
}

// Sound: early-bound lifetimes (not affected by HRTB coercion)
fn early_bound<'a>(x: &'a i32) -> &'a i32 { x }
fn test_early_bound() {
    let _f: fn(&i32) -> &i32 = early_bound;
}

// Sound: higher-ranked mapper introduces implied bounds from the type
// parameters to the bound lifetime (`K: 'a`, `V: 'a`), but not between two
// bound lifetimes. This pattern occurs in litemap.
type MapF<K, V> = for<'a> fn(&'a (K, V)) -> (&'a K, &'a V);

fn map_f<K, V>((k, v): &(K, V)) -> (&K, &V) { (k, v) }

fn test_litemap_style_mapper<K, V>(slice: &[(K, V)]) {
    let map_f: MapF<K, V> = map_f;
    let _ = slice.iter().map(map_f);
}

fn main() {
    test_no_lifetimes();
    test_preserved();
    test_early_bound();
}
