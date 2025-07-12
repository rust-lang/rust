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

//@ arg foo .index[] | select(.name == "Foo").inner.variant.kind?.struct.fields?
//@ jq .index[] | select(.name == "ews_0").id == $foo[0]
//@ jq .index[] | select(.name == "dik_1").id == $foo[1]
//@ jq .index[] | select(.name == "hsk_2").id == $foo[2]
//@ jq .index[] | select(.name == "djt_3").id == $foo[3]
//@ jq .index[] | select(.name == "jnr_4").id == $foo[4]
//@ jq .index[] | select(.name == "dfs_5").id == $foo[5]
//@ jq .index[] | select(.name == "bja_6").id == $foo[6]
//@ jq .index[] | select(.name == "lyc_7").id == $foo[7]
//@ jq .index[] | select(.name == "yqd_8").id == $foo[8]
//@ jq .index[] | select(.name == "vll_9").id == $foo[9]
