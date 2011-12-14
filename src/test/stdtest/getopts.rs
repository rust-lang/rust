import core::*;

use std;
import opt = std::getopts;

tag fail_type {
    argument_missing;
    unrecognized_option;
    option_missing;
    option_duplicated;
    unexpected_argument;
}

fn check_fail_type(f: opt::fail_, ft: fail_type) {
    alt f {
      opt::argument_missing(_) { assert (ft == argument_missing); }
      opt::unrecognized_option(_) { assert (ft == unrecognized_option); }
      opt::option_missing(_) { assert (ft == option_missing); }
      opt::option_duplicated(_) { assert (ft == option_duplicated); }
      opt::unexpected_argument(_) { assert (ft == unexpected_argument); }
      _ { fail; }
    }
}


// Tests for reqopt
#[test]
fn test_reqopt_long() {
    let args = ["--test=20"];
    let opts = [opt::reqopt("test")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::success(m) {
        assert (opt::opt_present(m, "test"));
        assert (opt::opt_str(m, "test") == "20");
      }
      _ { fail; }
    }
}

#[test]
fn test_reqopt_long_missing() {
    let args = ["blah"];
    let opts = [opt::reqopt("test")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::failure(f) { check_fail_type(f, option_missing); }
      _ { fail; }
    }
}

#[test]
fn test_reqopt_long_no_arg() {
    let args = ["--test"];
    let opts = [opt::reqopt("test")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::failure(f) { check_fail_type(f, argument_missing); }
      _ { fail; }
    }
}

#[test]
fn test_reqopt_long_multi() {
    let args = ["--test=20", "--test=30"];
    let opts = [opt::reqopt("test")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::failure(f) { check_fail_type(f, option_duplicated); }
      _ { fail; }
    }
}

#[test]
fn test_reqopt_short() {
    let args = ["-t", "20"];
    let opts = [opt::reqopt("t")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::success(m) {
        assert (opt::opt_present(m, "t"));
        assert (opt::opt_str(m, "t") == "20");
      }
      _ { fail; }
    }
}

#[test]
fn test_reqopt_short_missing() {
    let args = ["blah"];
    let opts = [opt::reqopt("t")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::failure(f) { check_fail_type(f, option_missing); }
      _ { fail; }
    }
}

#[test]
fn test_reqopt_short_no_arg() {
    let args = ["-t"];
    let opts = [opt::reqopt("t")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::failure(f) { check_fail_type(f, argument_missing); }
      _ { fail; }
    }
}

#[test]
fn test_reqopt_short_multi() {
    let args = ["-t", "20", "-t", "30"];
    let opts = [opt::reqopt("t")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::failure(f) { check_fail_type(f, option_duplicated); }
      _ { fail; }
    }
}


// Tests for optopt
#[test]
fn test_optopt_long() {
    let args = ["--test=20"];
    let opts = [opt::optopt("test")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::success(m) {
        assert (opt::opt_present(m, "test"));
        assert (opt::opt_str(m, "test") == "20");
      }
      _ { fail; }
    }
}

#[test]
fn test_optopt_long_missing() {
    let args = ["blah"];
    let opts = [opt::optopt("test")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::success(m) { assert (!opt::opt_present(m, "test")); }
      _ { fail; }
    }
}

#[test]
fn test_optopt_long_no_arg() {
    let args = ["--test"];
    let opts = [opt::optopt("test")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::failure(f) { check_fail_type(f, argument_missing); }
      _ { fail; }
    }
}

#[test]
fn test_optopt_long_multi() {
    let args = ["--test=20", "--test=30"];
    let opts = [opt::optopt("test")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::failure(f) { check_fail_type(f, option_duplicated); }
      _ { fail; }
    }
}

#[test]
fn test_optopt_short() {
    let args = ["-t", "20"];
    let opts = [opt::optopt("t")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::success(m) {
        assert (opt::opt_present(m, "t"));
        assert (opt::opt_str(m, "t") == "20");
      }
      _ { fail; }
    }
}

#[test]
fn test_optopt_short_missing() {
    let args = ["blah"];
    let opts = [opt::optopt("t")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::success(m) { assert (!opt::opt_present(m, "t")); }
      _ { fail; }
    }
}

#[test]
fn test_optopt_short_no_arg() {
    let args = ["-t"];
    let opts = [opt::optopt("t")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::failure(f) { check_fail_type(f, argument_missing); }
      _ { fail; }
    }
}

#[test]
fn test_optopt_short_multi() {
    let args = ["-t", "20", "-t", "30"];
    let opts = [opt::optopt("t")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::failure(f) { check_fail_type(f, option_duplicated); }
      _ { fail; }
    }
}


// Tests for optflag
#[test]
fn test_optflag_long() {
    let args = ["--test"];
    let opts = [opt::optflag("test")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::success(m) { assert (opt::opt_present(m, "test")); }
      _ { fail; }
    }
}

#[test]
fn test_optflag_long_missing() {
    let args = ["blah"];
    let opts = [opt::optflag("test")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::success(m) { assert (!opt::opt_present(m, "test")); }
      _ { fail; }
    }
}

#[test]
fn test_optflag_long_arg() {
    let args = ["--test=20"];
    let opts = [opt::optflag("test")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::failure(f) {
        log_err opt::fail_str(f);
        check_fail_type(f, unexpected_argument);
      }
      _ { fail; }
    }
}

#[test]
fn test_optflag_long_multi() {
    let args = ["--test", "--test"];
    let opts = [opt::optflag("test")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::failure(f) { check_fail_type(f, option_duplicated); }
      _ { fail; }
    }
}

#[test]
fn test_optflag_short() {
    let args = ["-t"];
    let opts = [opt::optflag("t")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::success(m) { assert (opt::opt_present(m, "t")); }
      _ { fail; }
    }
}

#[test]
fn test_optflag_short_missing() {
    let args = ["blah"];
    let opts = [opt::optflag("t")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::success(m) { assert (!opt::opt_present(m, "t")); }
      _ { fail; }
    }
}

#[test]
fn test_optflag_short_arg() {
    let args = ["-t", "20"];
    let opts = [opt::optflag("t")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::success(m) {
        // The next variable after the flag is just a free argument

        assert (m.free[0] == "20");
      }
      _ { fail; }
    }
}

#[test]
fn test_optflag_short_multi() {
    let args = ["-t", "-t"];
    let opts = [opt::optflag("t")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::failure(f) { check_fail_type(f, option_duplicated); }
      _ { fail; }
    }
}


// Tests for optmulti
#[test]
fn test_optmulti_long() {
    let args = ["--test=20"];
    let opts = [opt::optmulti("test")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::success(m) {
        assert (opt::opt_present(m, "test"));
        assert (opt::opt_str(m, "test") == "20");
      }
      _ { fail; }
    }
}

#[test]
fn test_optmulti_long_missing() {
    let args = ["blah"];
    let opts = [opt::optmulti("test")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::success(m) { assert (!opt::opt_present(m, "test")); }
      _ { fail; }
    }
}

#[test]
fn test_optmulti_long_no_arg() {
    let args = ["--test"];
    let opts = [opt::optmulti("test")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::failure(f) { check_fail_type(f, argument_missing); }
      _ { fail; }
    }
}

#[test]
fn test_optmulti_long_multi() {
    let args = ["--test=20", "--test=30"];
    let opts = [opt::optmulti("test")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::success(m) {
        assert (opt::opt_present(m, "test"));
        assert (opt::opt_str(m, "test") == "20");
        assert (opt::opt_strs(m, "test")[0] == "20");
        assert (opt::opt_strs(m, "test")[1] == "30");
      }
      _ { fail; }
    }
}

#[test]
fn test_optmulti_short() {
    let args = ["-t", "20"];
    let opts = [opt::optmulti("t")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::success(m) {
        assert (opt::opt_present(m, "t"));
        assert (opt::opt_str(m, "t") == "20");
      }
      _ { fail; }
    }
}

#[test]
fn test_optmulti_short_missing() {
    let args = ["blah"];
    let opts = [opt::optmulti("t")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::success(m) { assert (!opt::opt_present(m, "t")); }
      _ { fail; }
    }
}

#[test]
fn test_optmulti_short_no_arg() {
    let args = ["-t"];
    let opts = [opt::optmulti("t")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::failure(f) { check_fail_type(f, argument_missing); }
      _ { fail; }
    }
}

#[test]
fn test_optmulti_short_multi() {
    let args = ["-t", "20", "-t", "30"];
    let opts = [opt::optmulti("t")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::success(m) {
        assert (opt::opt_present(m, "t"));
        assert (opt::opt_str(m, "t") == "20");
        assert (opt::opt_strs(m, "t")[0] == "20");
        assert (opt::opt_strs(m, "t")[1] == "30");
      }
      _ { fail; }
    }
}

#[test]
fn test_unrecognized_option_long() {
    let args = ["--untest"];
    let opts = [opt::optmulti("t")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::failure(f) { check_fail_type(f, unrecognized_option); }
      _ { fail; }
    }
}

#[test]
fn test_unrecognized_option_short() {
    let args = ["-t"];
    let opts = [opt::optmulti("test")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::failure(f) { check_fail_type(f, unrecognized_option); }
      _ { fail; }
    }
}

#[test]
fn test_combined() {
    let args =
        ["prog", "free1", "-s", "20", "free2", "--flag", "--long=30", "-f",
         "-m", "40", "-m", "50"];
    let opts =
        [opt::optopt("s"), opt::optflag("flag"), opt::reqopt("long"),
         opt::optflag("f"), opt::optmulti("m"), opt::optopt("notpresent")];
    let rs = opt::getopts(args, opts);
    alt rs {
      opt::success(m) {
        assert (m.free[0] == "prog");
        assert (m.free[1] == "free1");
        assert (opt::opt_str(m, "s") == "20");
        assert (m.free[2] == "free2");
        assert (opt::opt_present(m, "flag"));
        assert (opt::opt_str(m, "long") == "30");
        assert (opt::opt_present(m, "f"));
        assert (opt::opt_strs(m, "m")[0] == "40");
        assert (opt::opt_strs(m, "m")[1] == "50");
        assert (!opt::opt_present(m, "notpresent"));
      }
      _ { fail; }
    }
}

