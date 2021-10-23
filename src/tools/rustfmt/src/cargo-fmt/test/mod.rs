use super::*;

mod message_format;
mod targets;

#[test]
fn default_options() {
    let empty: Vec<String> = vec![];
    let o = Opts::from_iter(&empty);
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
    let o = Opts::from_iter(&[
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
        Opts::clap()
            .get_matches_from_safe(&["test", "unexpected"])
            .is_err()
    );
}

#[test]
fn unexpected_flag() {
    assert!(
        Opts::clap()
            .get_matches_from_safe(&["test", "--flag"])
            .is_err()
    );
}

#[test]
fn mandatory_separator() {
    assert!(
        Opts::clap()
            .get_matches_from_safe(&["test", "--emit"])
            .is_err()
    );
    assert!(
        !Opts::clap()
            .get_matches_from_safe(&["test", "--", "--emit"])
            .is_err()
    );
}

#[test]
fn multiple_packages_one_by_one() {
    let o = Opts::from_iter(&[
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
    let o = Opts::from_iter(&[
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
    assert!(Opts::clap().get_matches_from_safe(&["test", "-p"]).is_err());
}

#[test]
fn empty_packages_2() {
    assert!(
        Opts::clap()
            .get_matches_from_safe(&["test", "-p", "--", "--check"])
            .is_err()
    );
}

#[test]
fn empty_packages_3() {
    assert!(
        Opts::clap()
            .get_matches_from_safe(&["test", "-p", "--verbose"])
            .is_err()
    );
}

#[test]
fn empty_packages_4() {
    assert!(
        Opts::clap()
            .get_matches_from_safe(&["test", "-p", "--check"])
            .is_err()
    );
}
