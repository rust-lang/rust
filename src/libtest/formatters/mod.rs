use super::*;

mod pretty;
mod json;
mod terse;

pub(crate) use self::pretty::PrettyFormatter;
pub(crate) use self::json::JsonFormatter;
pub(crate) use self::terse::TerseFormatter;

pub(crate) trait OutputFormatter {
    fn write_run_start(&mut self, test_count: usize) -> io::Result<()>;
    fn write_test_start(&mut self, desc: &TestDesc) -> io::Result<()>;
    fn write_timeout(&mut self, desc: &TestDesc) -> io::Result<()>;
    fn write_result(
        &mut self,
        desc: &TestDesc,
        result: &TestResult,
        stdout: &[u8],
    ) -> io::Result<()>;
    fn write_run_finish(&mut self, state: &ConsoleTestState) -> io::Result<bool>;
}
