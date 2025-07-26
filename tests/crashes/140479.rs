//@ known-bug: #140479
macro_rules! a { ( $( { $ [ $b:c ] } )) => ( $(${ concat(d, $b)} ))}
fn e() {
    a!({})
}
