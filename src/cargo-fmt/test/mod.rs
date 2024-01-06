use super::*;

mod message_format;
mod targets;

#[test]
fn default_options() {
    let empty: Vec<String> = vec![];
    let o = Opts::parse_from(&empty);
    assert_eq!(false, o.quiet);
    assert_eq!(false, o.verbose);
    assert_eq!(false, o.version);
    assert_eq!(false, o.check);
    assert_eq!(empty, o.packages);
    assert_eq!(empty, o.rustfmt_options);
    assert_eq!(false, o.format_all);
    assert_eq!(None, o.manifest_path);
    assert_eq!(None, o.message_format);
}

#[test]
fn good_options() {
    let o = Opts::parse_from([
        "test",
        "-q",
        "-p",
        "p1",
        "-p",
        "p2",
        "--message-format",
        "short",
        "--check",
        "--",
        "--edition",
        "2018",
    ]);
    assert_eq!(true, o.quiet);
    assert_eq!(false, o.verbose);
    assert_eq!(false, o.version);
    assert_eq!(true, o.check);
    assert_eq!(vec!["p1", "p2"], o.packages);
    assert_eq!(vec!["--edition", "2018"], o.rustfmt_options);
    assert_eq!(false, o.format_all);
    assert_eq!(Some(String::from("short")), o.message_format);
}

#[test]
fn unexpected_option() {
    assert!(
        Opts::command()
            .try_get_matches_from(["test", "unexpected"])
            .is_err()
    );
}

#[test]
fn unexpected_flag() {
    assert!(
        Opts::command()
            .try_get_matches_from(["test", "--flag"])
            .is_err()
    );
}

#[test]
fn mandatory_separator() {
    assert!(
        Opts::command()
            .try_get_matches_from(["test", "--emit"])
            .is_err()
    );
    assert!(
        Opts::command()
            .try_get_matches_from(["test", "--", "--emit"])
            .is_ok()
    );
}

#[test]
fn multiple_packages_one_by_one() {
    let o = Opts::parse_from([
        "test",
        "-p",
        "package1",
        "--package",
        "package2",
        "-p",
        "package3",
    ]);
    assert_eq!(3, o.packages.len());
}

#[test]
fn multiple_packages_grouped() {
    let o = Opts::parse_from([
        "test",
        "--package",
        "package1",
        "package2",
        "-p",
        "package3",
        "package4",
    ]);
    assert_eq!(4, o.packages.len());
}

#[test]
fn empty_packages_1() {
    assert!(
        Opts::command()
            .try_get_matches_from(["test", "-p"])
            .is_err()
    );
}

#[test]
fn empty_packages_2() {
    assert!(
        Opts::command()
            .try_get_matches_from(["test", "-p", "--", "--check"])
            .is_err()
    );
}

#[test]
fn empty_packages_3() {
    assert!(
        Opts::command()
            .try_get_matches_from(["test", "-p", "--verbose"])
            .is_err()
    );
}

#[test]
fn empty_packages_4() {
    assert!(
        Opts::command()
            .try_get_matches_from(["test", "-p", "--check"])
            .is_err()
    );
}
