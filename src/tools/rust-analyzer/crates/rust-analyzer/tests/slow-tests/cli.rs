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
        {"id":4,"type":"vertex","label":"range","start":{"line":0,"character":3},"end":{"line":0,"character":8}}
        {"id":5,"type":"vertex","label":"resultSet"}
        {"id":6,"type":"edge","label":"next","inV":5,"outV":4}
        {"id":7,"type":"vertex","label":"range","start":{"line":2,"character":13},"end":{"line":2,"character":43}}
        {"id":8,"type":"vertex","label":"resultSet"}
        {"id":9,"type":"edge","label":"next","inV":8,"outV":7}
        {"id":10,"type":"vertex","label":"range","start":{"line":8,"character":0},"end":{"line":8,"character":30}}
        {"id":11,"type":"edge","label":"next","inV":8,"outV":10}
        {"id":12,"type":"vertex","label":"range","start":{"line":8,"character":32},"end":{"line":8,"character":39}}
        {"id":13,"type":"vertex","label":"resultSet"}
        {"id":14,"type":"edge","label":"next","inV":13,"outV":12}
        {"id":15,"type":"vertex","label":"range","start":{"line":9,"character":4},"end":{"line":9,"character":9}}
        {"id":16,"type":"vertex","label":"resultSet"}
        {"id":17,"type":"edge","label":"next","inV":16,"outV":15}
        {"id":18,"type":"vertex","label":"range","start":{"line":10,"character":8},"end":{"line":10,"character":13}}
        {"id":19,"type":"vertex","label":"resultSet"}
        {"id":20,"type":"edge","label":"next","inV":19,"outV":18}
        {"id":21,"type":"vertex","label":"range","start":{"line":11,"character":4},"end":{"line":11,"character":34}}
        {"id":22,"type":"edge","label":"next","inV":8,"outV":21}
        {"id":23,"type":"vertex","label":"range","start":{"line":11,"character":36},"end":{"line":11,"character":43}}
        {"id":24,"type":"vertex","label":"resultSet"}
        {"id":25,"type":"edge","label":"next","inV":24,"outV":23}
        {"id":26,"type":"edge","label":"contains","inVs":[4,7,10,12,15,18,21,23],"outV":1}
        {"id":27,"type":"vertex","label":"hoverResult","result":{"contents":{"kind":"markdown","value":"\n```rust\n#[allow]\n```\n\n---\n\nValid forms are:\n\n* \\#\\[allow(lint1, lint2, ..., /\\*opt\\*/ reason = \"...\")\\]"}}}
        {"id":28,"type":"edge","label":"textDocument/hover","inV":27,"outV":5}
        {"id":29,"type":"vertex","label":"referenceResult"}
        {"id":30,"type":"edge","label":"textDocument/references","inV":29,"outV":5}
        {"id":31,"type":"edge","label":"item","document":1,"property":"references","inVs":[4],"outV":29}
        {"id":32,"type":"vertex","label":"hoverResult","result":{"contents":{"kind":"markdown","value":"\n```rust\nfoo\n```\n\n```rust\nmacro_rules! generate_const_from_identifier\n```"}}}
        {"id":33,"type":"edge","label":"textDocument/hover","inV":32,"outV":8}
        {"id":34,"type":"vertex","label":"packageInformation","name":"foo","manager":"cargo","version":"0.0.0"}
        {"id":35,"type":"vertex","label":"moniker","scheme":"rust-analyzer","identifier":"foo::generate_const_from_identifier","unique":"scheme","kind":"export"}
        {"id":36,"type":"edge","label":"packageInformation","inV":34,"outV":35}
        {"id":37,"type":"edge","label":"moniker","inV":35,"outV":8}
        {"id":38,"type":"vertex","label":"definitionResult"}
        {"id":39,"type":"edge","label":"item","document":1,"inVs":[7],"outV":38}
        {"id":40,"type":"edge","label":"textDocument/definition","inV":38,"outV":8}
        {"id":41,"type":"vertex","label":"referenceResult"}
        {"id":42,"type":"edge","label":"textDocument/references","inV":41,"outV":8}
        {"id":43,"type":"edge","label":"item","document":1,"property":"definitions","inVs":[7],"outV":41}
        {"id":44,"type":"edge","label":"item","document":1,"property":"references","inVs":[10,21],"outV":41}
        {"id":45,"type":"vertex","label":"hoverResult","result":{"contents":{"kind":"markdown","value":"\n```rust\nfoo\n```\n\n```rust\nconst REQ_001: &str = \"encoded_data\"\n```"}}}
        {"id":46,"type":"edge","label":"textDocument/hover","inV":45,"outV":13}
        {"id":47,"type":"vertex","label":"moniker","scheme":"rust-analyzer","identifier":"foo::REQ_001","unique":"scheme","kind":"export"}
        {"id":48,"type":"edge","label":"packageInformation","inV":34,"outV":47}
        {"id":49,"type":"edge","label":"moniker","inV":47,"outV":13}
        {"id":50,"type":"vertex","label":"definitionResult"}
        {"id":51,"type":"edge","label":"item","document":1,"inVs":[12],"outV":50}
        {"id":52,"type":"edge","label":"textDocument/definition","inV":50,"outV":13}
        {"id":53,"type":"vertex","label":"referenceResult"}
        {"id":54,"type":"edge","label":"textDocument/references","inV":53,"outV":13}
        {"id":55,"type":"edge","label":"item","document":1,"property":"definitions","inVs":[12],"outV":53}
        {"id":56,"type":"vertex","label":"hoverResult","result":{"contents":{"kind":"markdown","value":"\n```rust\nfoo\n```\n\n```rust\nmod tests\n```"}}}
        {"id":57,"type":"edge","label":"textDocument/hover","inV":56,"outV":16}
        {"id":58,"type":"vertex","label":"moniker","scheme":"rust-analyzer","identifier":"foo::tests","unique":"scheme","kind":"export"}
        {"id":59,"type":"edge","label":"packageInformation","inV":34,"outV":58}
        {"id":60,"type":"edge","label":"moniker","inV":58,"outV":16}
        {"id":61,"type":"vertex","label":"definitionResult"}
        {"id":62,"type":"edge","label":"item","document":1,"inVs":[15],"outV":61}
        {"id":63,"type":"edge","label":"textDocument/definition","inV":61,"outV":16}
        {"id":64,"type":"vertex","label":"referenceResult"}
        {"id":65,"type":"edge","label":"textDocument/references","inV":64,"outV":16}
        {"id":66,"type":"edge","label":"item","document":1,"property":"definitions","inVs":[15],"outV":64}
        {"id":67,"type":"vertex","label":"hoverResult","result":{"contents":{"kind":"markdown","value":"\n```rust\nextern crate foo\n```"}}}
        {"id":68,"type":"edge","label":"textDocument/hover","inV":67,"outV":19}
        {"id":69,"type":"vertex","label":"definitionResult"}
        {"id":70,"type":"vertex","label":"range","start":{"line":0,"character":0},"end":{"line":13,"character":0}}
        {"id":71,"type":"edge","label":"contains","inVs":[70],"outV":1}
        {"id":72,"type":"edge","label":"item","document":1,"inVs":[70],"outV":69}
        {"id":73,"type":"edge","label":"textDocument/definition","inV":69,"outV":19}
        {"id":74,"type":"vertex","label":"referenceResult"}
        {"id":75,"type":"edge","label":"textDocument/references","inV":74,"outV":19}
        {"id":76,"type":"edge","label":"item","document":1,"property":"references","inVs":[18],"outV":74}
        {"id":77,"type":"vertex","label":"hoverResult","result":{"contents":{"kind":"markdown","value":"\n```rust\nfoo::tests\n```\n\n```rust\nconst REQ_002: &str = \"encoded_data\"\n```"}}}
        {"id":78,"type":"edge","label":"textDocument/hover","inV":77,"outV":24}
        {"id":79,"type":"vertex","label":"moniker","scheme":"rust-analyzer","identifier":"foo::tests::REQ_002","unique":"scheme","kind":"export"}
        {"id":80,"type":"edge","label":"packageInformation","inV":34,"outV":79}
        {"id":81,"type":"edge","label":"moniker","inV":79,"outV":24}
        {"id":82,"type":"vertex","label":"definitionResult"}
        {"id":83,"type":"edge","label":"item","document":1,"inVs":[23],"outV":82}
        {"id":84,"type":"edge","label":"textDocument/definition","inV":82,"outV":24}
        {"id":85,"type":"vertex","label":"referenceResult"}
        {"id":86,"type":"edge","label":"textDocument/references","inV":85,"outV":24}
        {"id":87,"type":"edge","label":"item","document":1,"property":"definitions","inVs":[23],"outV":85}
    "#]].assert_eq(stdout);
}
