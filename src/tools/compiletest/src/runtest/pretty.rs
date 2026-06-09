use std::fs;

use super::{ProcRes, ReadFrom, TestCx};

impl TestCx<'_> {
    pub(super) fn run_pretty_test(&self) {
        if self.props.pp_exact.is_some() {
            self.logv("testing for exact pretty-printing");
        } else {
            self.logv("testing for converging pretty-printing");
        }

        let rounds = match self.props.pp_exact {
            Some(_) => 1,
            None => 2,
        };

        let src = fs::read_to_string(&self.testpaths.file).unwrap();
        let mut srcs = vec![src];

        let mut round = 0;
        while round < rounds {
            self.logv(format_args!("pretty-printing round {round} revision {:?}", self.revision));
            let read_from =
                if round == 0 { ReadFrom::Path } else { ReadFrom::Stdin(srcs[round].to_owned()) };

            let proc_res = self.print_source(read_from, &self.props.pretty_mode);
            if !proc_res.status.success() {
                self.fatal_proc_rec(
                    &format!(
                        "pretty-printing failed in round {} revision {:?}",
                        round, self.revision
                    ),
                    &proc_res,
                );
            }

            let ProcRes { stdout, .. } = proc_res;
            srcs.push(stdout);
            round += 1;
        }

        let mut expected = match self.props.pp_exact {
            Some(ref file) => {
                let filepath = self.testpaths.file.parent().unwrap().join(file);
                fs::read_to_string(&filepath).unwrap()
            }
            None => srcs[srcs.len() - 2].clone(),
        };
        let mut actual = srcs[srcs.len() - 1].clone();

        if self.props.pp_exact.is_some() {
            // Now we have to care about line endings
            let cr = "\r".to_owned();
            actual = actual.replace(&cr, "");
            expected = expected.replace(&cr, "");
        }

        if !self.config.bless {
            self.compare_source(&expected, &actual);
        } else if expected != actual {
            let filepath_buf;
            let filepath = match &self.props.pp_exact {
                Some(file) => {
                    filepath_buf = self.testpaths.file.parent().unwrap().join(file);
                    &filepath_buf
                }
                None => &self.testpaths.file,
            };
            fs::write(filepath, &actual).unwrap();
        }

        // If we're only making sure that the output matches then just stop here
        if self.props.pretty_compare_only {
            return;
        }

        // Finally, let's make sure it actually appears to remain valid code
        let proc_res = self.typecheck_source(actual);
        if !proc_res.status.success() {
            self.fatal_proc_rec("pretty-printed source does not typecheck", &proc_res);
        }
    }
}
