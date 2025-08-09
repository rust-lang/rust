use crate::config::ShouldMerge;
use crate::html::render::ordered_json::{EscapedJson, OrderedJson};
use crate::html::render::sorted_template::{Html, SortedTemplate};
use crate::html::render::write_shared::*;

#[test]
fn hack_external_crate_names() {
    let path = tempfile::TempDir::new().unwrap();
    let path = path.path();
    let crates = hack_get_external_crate_names(&path, "").unwrap();
    assert!(crates.is_empty());
    fs::write(path.join("crates.js"), r#"window.ALL_CRATES = ["a","b","c"];"#).unwrap();
    let crates = hack_get_external_crate_names(&path, "").unwrap();
    assert_eq!(crates, ["a".to_string(), "b".to_string(), "c".to_string()]);
}

fn but_last_line(s: &str) -> &str {
    let (before, _) = s.rsplit_once("\n").unwrap();
    before
}

#[test]
fn sources_template() {
    let mut template = SourcesPart::blank();
    assert_eq!(but_last_line(&template.to_string()), r"createSrcSidebar('[]');");
    template.append(EscapedJson::from(OrderedJson::serialize("u").unwrap()).to_string());
    assert_eq!(but_last_line(&template.to_string()), r#"createSrcSidebar('["u"]');"#);
    template.append(EscapedJson::from(OrderedJson::serialize("v").unwrap()).to_string());
    assert_eq!(but_last_line(&template.to_string()), r#"createSrcSidebar('["u","v"]');"#);
}

#[test]
fn all_crates_template() {
    let mut template = AllCratesPart::blank();
    assert_eq!(but_last_line(&template.to_string()), r"window.ALL_CRATES = [];");
    template.append(EscapedJson::from(OrderedJson::serialize("b").unwrap()).to_string());
    assert_eq!(but_last_line(&template.to_string()), r#"window.ALL_CRATES = ["b"];"#);
    template.append(EscapedJson::from(OrderedJson::serialize("a").unwrap()).to_string());
    assert_eq!(but_last_line(&template.to_string()), r#"window.ALL_CRATES = ["a","b"];"#);
}

#[test]
fn all_crates_parts() {
    let parts = AllCratesPart::get(OrderedJson::serialize("crate").unwrap(), "").unwrap();
    assert_eq!(&parts.parts[0].0, Path::new("crates.js"));
    assert_eq!(&parts.parts[0].1.to_string(), r#""crate""#);
}

#[test]
fn crates_index_part() {
    let external_crates = ["bar".to_string(), "baz".to_string()];
    let mut parts = CratesIndexPart::get("foo", &external_crates).unwrap();
    parts.parts.sort_by(|a, b| a.1.to_string().cmp(&b.1.to_string()));

    assert_eq!(&parts.parts[0].0, Path::new("index.html"));
    assert_eq!(&parts.parts[0].1.to_string(), r#"<li><a href="bar/index.html">bar</a></li>"#);

    assert_eq!(&parts.parts[1].0, Path::new("index.html"));
    assert_eq!(&parts.parts[1].1.to_string(), r#"<li><a href="baz/index.html">baz</a></li>"#);

    assert_eq!(&parts.parts[2].0, Path::new("index.html"));
    assert_eq!(&parts.parts[2].1.to_string(), r#"<li><a href="foo/index.html">foo</a></li>"#);
}

#[test]
fn trait_alias_template() {
    let mut template = TraitAliasPart::blank();
    assert_eq!(
        but_last_line(&template.to_string()),
        r#"(function() {
    var implementors = Object.fromEntries([]);
    if (window.register_implementors) {
        window.register_implementors(implementors);
    } else {
        window.pending_implementors = implementors;
    }
})()"#,
    );
    template.append(OrderedJson::serialize(["a"]).unwrap().to_string());
    assert_eq!(
        but_last_line(&template.to_string()),
        r#"(function() {
    var implementors = Object.fromEntries([["a"]]);
    if (window.register_implementors) {
        window.register_implementors(implementors);
    } else {
        window.pending_implementors = implementors;
    }
})()"#,
    );
    template.append(OrderedJson::serialize(["b"]).unwrap().to_string());
    assert_eq!(
        but_last_line(&template.to_string()),
        r#"(function() {
    var implementors = Object.fromEntries([["a"],["b"]]);
    if (window.register_implementors) {
        window.register_implementors(implementors);
    } else {
        window.pending_implementors = implementors;
    }
})()"#,
    );
}

#[test]
fn type_alias_template() {
    let mut template = TypeAliasPart::blank();
    assert_eq!(
        but_last_line(&template.to_string()),
        r#"(function() {
    var type_impls = Object.fromEntries([]);
    if (window.register_type_impls) {
        window.register_type_impls(type_impls);
    } else {
        window.pending_type_impls = type_impls;
    }
})()"#,
    );
    template.append(OrderedJson::serialize(["a"]).unwrap().to_string());
    assert_eq!(
        but_last_line(&template.to_string()),
        r#"(function() {
    var type_impls = Object.fromEntries([["a"]]);
    if (window.register_type_impls) {
        window.register_type_impls(type_impls);
    } else {
        window.pending_type_impls = type_impls;
    }
})()"#,
    );
    template.append(OrderedJson::serialize(["b"]).unwrap().to_string());
    assert_eq!(
        but_last_line(&template.to_string()),
        r#"(function() {
    var type_impls = Object.fromEntries([["a"],["b"]]);
    if (window.register_type_impls) {
        window.register_type_impls(type_impls);
    } else {
        window.pending_type_impls = type_impls;
    }
})()"#,
    );
}

#[test]
fn read_template_test() {
    let path = tempfile::TempDir::new().unwrap();
    let path = path.path().join("file.html");
    let make_blank = || SortedTemplate::<Html>::from_before_after("<div>", "</div>");

    let should_merge = ShouldMerge { read_rendered_cci: true, write_rendered_cci: true };
    let template = read_template_or_blank(make_blank, &path, &should_merge).unwrap();
    assert_eq!(but_last_line(&template.to_string()), "<div></div>");
    fs::write(&path, template.to_string()).unwrap();
    let mut template = read_template_or_blank(make_blank, &path, &should_merge).unwrap();
    template.append("<img/>".to_string());
    fs::write(&path, template.to_string()).unwrap();
    let mut template = read_template_or_blank(make_blank, &path, &should_merge).unwrap();
    template.append("<br/>".to_string());
    fs::write(&path, template.to_string()).unwrap();
    let template = read_template_or_blank(make_blank, &path, &should_merge).unwrap();

    assert_eq!(but_last_line(&template.to_string()), "<div><br/><img/></div>");
}
