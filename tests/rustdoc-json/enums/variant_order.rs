// Check that the order of variants is preserved.

pub enum Foo {
    // Important: random prefixes are used here to ensure that
    // sorting fields by name would cause this test to fail.
    Ews0,
    Dik1,
    Hsk2,
    Djt3,
    Jnr4,
    Dfs5,
    Bja6,
    Lyc7,
    Yqd8,
    Vll9,
}

//@ arg foo .index[] | select(.name == "Foo").inner.enum.variants?
//@ jq .index[] | select(.name == "Ews0").id == $foo[0]
//@ jq .index[] | select(.name == "Dik1").id == $foo[1]
//@ jq .index[] | select(.name == "Hsk2").id == $foo[2]
//@ jq .index[] | select(.name == "Djt3").id == $foo[3]
//@ jq .index[] | select(.name == "Jnr4").id == $foo[4]
//@ jq .index[] | select(.name == "Dfs5").id == $foo[5]
//@ jq .index[] | select(.name == "Bja6").id == $foo[6]
//@ jq .index[] | select(.name == "Lyc7").id == $foo[7]
//@ jq .index[] | select(.name == "Yqd8").id == $foo[8]
//@ jq .index[] | select(.name == "Vll9").id == $foo[9]
