use expect_test::expect;
use test_utils::skip_slow_tests;

use crate::support::Project;

// If you choose to change the test fixture here, please inform the ferrocene/needy maintainers by
// opening an issue at https://github.com/ferrocene/needy as the tool relies on specific token
// mapping behavior.
#[test]
fn lsif_contains_generated_constant() {
    if skip_slow_tests() {
        return;
    }

    let stdout = Project::with_fixture(
        r#"
//- /Cargo.toml
[package]
name = "foo"
version = "0.0.0"

//- /src/lib.rs
#![allow(unused)]

macro_rules! generate_const_from_identifier(
    ($id:ident) => (
        const _: () = { const $id: &str = "encoded_data"; };
    )
);

generate_const_from_identifier!(REQ_001);
mod tests {
    use super::*;
    generate_const_from_identifier!(REQ_002);
}
"#,
    )
    .root("foo")
    .run_lsif();
    let n = stdout.find(r#"{"id":2,"#).unwrap();
    // the first 2 entries contain paths that are not stable
    let stdout = &stdout[n..];
    expect![[r#"
        {"id":2,"type":"vertex","label":"foldingRangeResult","result":[{"startLine":2,"startCharacter":43,"endLine":6,"endCharacter":1},{"startLine":3,"startCharacter":19,"endLine":5,"endCharacter":5},{"startLine":9,"startCharacter":10,"endLine":12,"endCharacter":1}]}
        {"id":3,"type":"edge","label":"textDocument/foldingRange","inV":2,"outV":1}
        {"id":4,"type":"vertex","label":"range","start":{"line":0,"character":0},"end":{"line":13,"character":0}}
        {"id":5,"type":"vertex","label":"resultSet"}
        {"id":6,"type":"edge","label":"next","inV":5,"outV":4}
        {"id":7,"type":"vertex","label":"range","start":{"line":0,"character":3},"end":{"line":0,"character":8}}
        {"id":8,"type":"vertex","label":"resultSet"}
        {"id":9,"type":"edge","label":"next","inV":8,"outV":7}
        {"id":10,"type":"vertex","label":"range","start":{"line":2,"character":13},"end":{"line":2,"character":43}}
        {"id":11,"type":"vertex","label":"resultSet"}
        {"id":12,"type":"edge","label":"next","inV":11,"outV":10}
        {"id":13,"type":"vertex","label":"range","start":{"line":8,"character":0},"end":{"line":8,"character":30}}
        {"id":14,"type":"edge","label":"next","inV":11,"outV":13}
        {"id":15,"type":"vertex","label":"range","start":{"line":8,"character":32},"end":{"line":8,"character":39}}
        {"id":16,"type":"vertex","label":"resultSet"}
        {"id":17,"type":"edge","label":"next","inV":16,"outV":15}
        {"id":18,"type":"vertex","label":"range","start":{"line":9,"character":4},"end":{"line":9,"character":9}}
        {"id":19,"type":"vertex","label":"resultSet"}
        {"id":20,"type":"edge","label":"next","inV":19,"outV":18}
        {"id":21,"type":"vertex","label":"range","start":{"line":10,"character":8},"end":{"line":10,"character":13}}
        {"id":22,"type":"edge","label":"next","inV":5,"outV":21}
        {"id":23,"type":"vertex","label":"range","start":{"line":11,"character":4},"end":{"line":11,"character":34}}
        {"id":24,"type":"edge","label":"next","inV":11,"outV":23}
        {"id":25,"type":"vertex","label":"range","start":{"line":11,"character":36},"end":{"line":11,"character":43}}
        {"id":26,"type":"vertex","label":"resultSet"}
        {"id":27,"type":"edge","label":"next","inV":26,"outV":25}
        {"id":28,"type":"edge","label":"contains","inVs":[4,7,10,13,15,18,21,23,25],"outV":1}
        {"id":29,"type":"vertex","label":"hoverResult","result":{"contents":{"kind":"markdown","value":"\n```rust\nextern crate foo\n```"}}}
        {"id":30,"type":"edge","label":"textDocument/hover","inV":29,"outV":5}
        {"id":31,"type":"vertex","label":"packageInformation","name":"foo","manager":"cargo","version":"0.0.0"}
        {"id":32,"type":"vertex","label":"moniker","scheme":"rust-analyzer","identifier":"foo::crate","unique":"scheme","kind":"export"}
        {"id":33,"type":"edge","label":"packageInformation","inV":31,"outV":32}
        {"id":34,"type":"edge","label":"moniker","inV":32,"outV":5}
        {"id":35,"type":"vertex","label":"definitionResult"}
        {"id":36,"type":"edge","label":"item","document":1,"inVs":[4],"outV":35}
        {"id":37,"type":"edge","label":"textDocument/definition","inV":35,"outV":5}
        {"id":38,"type":"vertex","label":"referenceResult"}
        {"id":39,"type":"edge","label":"textDocument/references","inV":38,"outV":5}
        {"id":40,"type":"edge","label":"item","document":1,"property":"definitions","inVs":[4],"outV":38}
        {"id":41,"type":"edge","label":"item","document":1,"property":"references","inVs":[21],"outV":38}
        {"id":42,"type":"vertex","label":"hoverResult","result":{"contents":{"kind":"markdown","value":"\n```rust\n#[allow]\n```\n\n---\n\nValid forms are:\n\n* \\#\\[allow(lint1, lint2, ..., /\\*opt\\*/ reason = \"...\")\\]"}}}
        {"id":43,"type":"edge","label":"textDocument/hover","inV":42,"outV":8}
        {"id":44,"type":"vertex","label":"referenceResult"}
        {"id":45,"type":"edge","label":"textDocument/references","inV":44,"outV":8}
        {"id":46,"type":"edge","label":"item","document":1,"property":"references","inVs":[7],"outV":44}
        {"id":47,"type":"vertex","label":"hoverResult","result":{"contents":{"kind":"markdown","value":"\n```rust\nfoo\n```\n\n```rust\nmacro_rules! generate_const_from_identifier\n```"}}}
        {"id":48,"type":"edge","label":"textDocument/hover","inV":47,"outV":11}
        {"id":49,"type":"vertex","label":"moniker","scheme":"rust-analyzer","identifier":"foo::generate_const_from_identifier","unique":"scheme","kind":"export"}
        {"id":50,"type":"edge","label":"packageInformation","inV":31,"outV":49}
        {"id":51,"type":"edge","label":"moniker","inV":49,"outV":11}
        {"id":52,"type":"vertex","label":"definitionResult"}
        {"id":53,"type":"edge","label":"item","document":1,"inVs":[10],"outV":52}
        {"id":54,"type":"edge","label":"textDocument/definition","inV":52,"outV":11}
        {"id":55,"type":"vertex","label":"referenceResult"}
        {"id":56,"type":"edge","label":"textDocument/references","inV":55,"outV":11}
        {"id":57,"type":"edge","label":"item","document":1,"property":"definitions","inVs":[10],"outV":55}
        {"id":58,"type":"edge","label":"item","document":1,"property":"references","inVs":[13,23],"outV":55}
        {"id":59,"type":"vertex","label":"hoverResult","result":{"contents":{"kind":"markdown","value":"\n```rust\nfoo\n```\n\n```rust\nconst REQ_001: &str = \"encoded_data\"\n```"}}}
        {"id":60,"type":"edge","label":"textDocument/hover","inV":59,"outV":16}
        {"id":61,"type":"vertex","label":"moniker","scheme":"rust-analyzer","identifier":"foo::REQ_001","unique":"scheme","kind":"export"}
        {"id":62,"type":"edge","label":"packageInformation","inV":31,"outV":61}
        {"id":63,"type":"edge","label":"moniker","inV":61,"outV":16}
        {"id":64,"type":"vertex","label":"definitionResult"}
        {"id":65,"type":"edge","label":"item","document":1,"inVs":[15],"outV":64}
        {"id":66,"type":"edge","label":"textDocument/definition","inV":64,"outV":16}
        {"id":67,"type":"vertex","label":"referenceResult"}
        {"id":68,"type":"edge","label":"textDocument/references","inV":67,"outV":16}
        {"id":69,"type":"edge","label":"item","document":1,"property":"definitions","inVs":[15],"outV":67}
        {"id":70,"type":"vertex","label":"hoverResult","result":{"contents":{"kind":"markdown","value":"\n```rust\nfoo\n```\n\n```rust\nmod tests\n```"}}}
        {"id":71,"type":"edge","label":"textDocument/hover","inV":70,"outV":19}
        {"id":72,"type":"vertex","label":"moniker","scheme":"rust-analyzer","identifier":"foo::tests","unique":"scheme","kind":"export"}
        {"id":73,"type":"edge","label":"packageInformation","inV":31,"outV":72}
        {"id":74,"type":"edge","label":"moniker","inV":72,"outV":19}
        {"id":75,"type":"vertex","label":"definitionResult"}
        {"id":76,"type":"edge","label":"item","document":1,"inVs":[18],"outV":75}
        {"id":77,"type":"edge","label":"textDocument/definition","inV":75,"outV":19}
        {"id":78,"type":"vertex","label":"referenceResult"}
        {"id":79,"type":"edge","label":"textDocument/references","inV":78,"outV":19}
        {"id":80,"type":"edge","label":"item","document":1,"property":"definitions","inVs":[18],"outV":78}
        {"id":81,"type":"vertex","label":"hoverResult","result":{"contents":{"kind":"markdown","value":"\n```rust\nfoo::tests\n```\n\n```rust\nconst REQ_002: &str = \"encoded_data\"\n```"}}}
        {"id":82,"type":"edge","label":"textDocument/hover","inV":81,"outV":26}
        {"id":83,"type":"vertex","label":"moniker","scheme":"rust-analyzer","identifier":"foo::tests::REQ_002","unique":"scheme","kind":"export"}
        {"id":84,"type":"edge","label":"packageInformation","inV":31,"outV":83}
        {"id":85,"type":"edge","label":"moniker","inV":83,"outV":26}
        {"id":86,"type":"vertex","label":"definitionResult"}
        {"id":87,"type":"edge","label":"item","document":1,"inVs":[25],"outV":86}
        {"id":88,"type":"edge","label":"textDocument/definition","inV":86,"outV":26}
        {"id":89,"type":"vertex","label":"referenceResult"}
        {"id":90,"type":"edge","label":"textDocument/references","inV":89,"outV":26}
        {"id":91,"type":"edge","label":"item","document":1,"property":"definitions","inVs":[25],"outV":89}
    "#]].assert_eq(stdout);
}
