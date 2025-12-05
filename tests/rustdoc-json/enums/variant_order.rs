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

//@ set 0 = '$.index[?(@.name == "Ews0")].id'
//@ set 1 = '$.index[?(@.name == "Dik1")].id'
//@ set 2 = '$.index[?(@.name == "Hsk2")].id'
//@ set 3 = '$.index[?(@.name == "Djt3")].id'
//@ set 4 = '$.index[?(@.name == "Jnr4")].id'
//@ set 5 = '$.index[?(@.name == "Dfs5")].id'
//@ set 6 = '$.index[?(@.name == "Bja6")].id'
//@ set 7 = '$.index[?(@.name == "Lyc7")].id'
//@ set 8 = '$.index[?(@.name == "Yqd8")].id'
//@ set 9 = '$.index[?(@.name == "Vll9")].id'

//@ is '$.index[?(@.name == "Foo")].inner.enum.variants[0]' $0
//@ is '$.index[?(@.name == "Foo")].inner.enum.variants[1]' $1
//@ is '$.index[?(@.name == "Foo")].inner.enum.variants[2]' $2
//@ is '$.index[?(@.name == "Foo")].inner.enum.variants[3]' $3
//@ is '$.index[?(@.name == "Foo")].inner.enum.variants[4]' $4
//@ is '$.index[?(@.name == "Foo")].inner.enum.variants[5]' $5
//@ is '$.index[?(@.name == "Foo")].inner.enum.variants[6]' $6
//@ is '$.index[?(@.name == "Foo")].inner.enum.variants[7]' $7
//@ is '$.index[?(@.name == "Foo")].inner.enum.variants[8]' $8
//@ is '$.index[?(@.name == "Foo")].inner.enum.variants[9]' $9
