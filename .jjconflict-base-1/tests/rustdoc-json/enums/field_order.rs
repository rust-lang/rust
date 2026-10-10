// Check that the order of fields is preserved.

pub enum Whatever {
    Foo {
        // Important: random prefixes are used here to ensure that
        // sorting fields by name would cause this test to fail.
        ews_0: i32,
        dik_1: i32,
        hsk_2: i32,
        djt_3: i32,
        jnr_4: i32,
        dfs_5: i32,
        bja_6: i32,
        lyc_7: i32,
        yqd_8: i32,
        vll_9: i32,
    },
}

//@ set 0 = '$.index[?(@.name == "ews_0")].id'
//@ set 1 = '$.index[?(@.name == "dik_1")].id'
//@ set 2 = '$.index[?(@.name == "hsk_2")].id'
//@ set 3 = '$.index[?(@.name == "djt_3")].id'
//@ set 4 = '$.index[?(@.name == "jnr_4")].id'
//@ set 5 = '$.index[?(@.name == "dfs_5")].id'
//@ set 6 = '$.index[?(@.name == "bja_6")].id'
//@ set 7 = '$.index[?(@.name == "lyc_7")].id'
//@ set 8 = '$.index[?(@.name == "yqd_8")].id'
//@ set 9 = '$.index[?(@.name == "vll_9")].id'

//@ is '$.index[?(@.name == "Foo")].inner.variant.kind.struct.fields[0]' $0
//@ is '$.index[?(@.name == "Foo")].inner.variant.kind.struct.fields[1]' $1
//@ is '$.index[?(@.name == "Foo")].inner.variant.kind.struct.fields[2]' $2
//@ is '$.index[?(@.name == "Foo")].inner.variant.kind.struct.fields[3]' $3
//@ is '$.index[?(@.name == "Foo")].inner.variant.kind.struct.fields[4]' $4
//@ is '$.index[?(@.name == "Foo")].inner.variant.kind.struct.fields[5]' $5
//@ is '$.index[?(@.name == "Foo")].inner.variant.kind.struct.fields[6]' $6
//@ is '$.index[?(@.name == "Foo")].inner.variant.kind.struct.fields[7]' $7
//@ is '$.index[?(@.name == "Foo")].inner.variant.kind.struct.fields[8]' $8
//@ is '$.index[?(@.name == "Foo")].inner.variant.kind.struct.fields[9]' $9
