//! Module containing different events that can occur
//! during tests execution process.

use super::test_result::TestResult;
use super::time::TestExecTime;
use super::types::TestDesc;

#[derive(Debug, Clone)]
pub struct CompletedTest {
    pub desc: TestDesc,
    pub result: TestResult,
    pub exec_time: Option<TestExecTime>,
    pub stdout: Vec<u8>,
}

impl CompletedTest {
    pub fn new(
        desc: TestDesc,
        result: TestResult,
        exec_time: Option<TestExecTime>,
        stdout: Vec<u8>,
    ) -> Self {
        Self { desc, result, exec_time, stdout }
    }
}

#[derive(Debug, Clone)]
pub enum TestEvent {
    TeFiltered(Vec<TestDesc>),
    TeWait(TestDesc),
    TeResult(CompletedTest),
    TeTimeout(TestDesc),
    TeFilteredOut(usize),
}
