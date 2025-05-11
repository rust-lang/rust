use std::{
    collections::BTreeMap,
    fs,
    io::Write as _,
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::format_err;
use xshell::{Shell, cmd};

use crate::flags::{self, MeasurementType};

type Unit = String;

impl flags::Metrics {
    pub(crate) fn run(self, sh: &Shell) -> anyhow::Result<()> {
        let mut metrics = Metrics::new(sh)?;
        if !Path::new("./target/rustc-perf").exists() {
            sh.create_dir("./target/rustc-perf")?;
            cmd!(sh, "git clone https://github.com/rust-lang/rustc-perf.git ./target/rustc-perf")
                .run()?;
        }
        {
            let _d = sh.push_dir("./target/rustc-perf");
            let revision = &metrics.perf_revision;
            cmd!(sh, "git reset --hard {revision}").run()?;
        }

        let _env = sh.push_env("RA_METRICS", "1");

        let name = match &self.measurement_type {
            Some(ms) => {
                let name = ms.as_ref();
                match ms {
                    MeasurementType::Build => {
                        metrics.measure_build(sh)?;
                    }
                    MeasurementType::RustcTests => {
                        metrics.measure_rustc_tests(sh)?;
                    }
                    MeasurementType::AnalyzeSelf => {
                        metrics.measure_analysis_stats_self(sh)?;
                    }
                    MeasurementType::AnalyzeRipgrep
                    | MeasurementType::AnalyzeWebRender
                    | MeasurementType::AnalyzeDiesel
                    | MeasurementType::AnalyzeHyper => {
                        metrics.measure_analysis_stats(sh, name)?;
                    }
                };
                name
            }
            None => {
                metrics.measure_build(sh)?;
                metrics.measure_rustc_tests(sh)?;
                metrics.measure_analysis_stats_self(sh)?;
                metrics.measure_analysis_stats(sh, MeasurementType::AnalyzeRipgrep.as_ref())?;
                metrics.measure_analysis_stats(sh, MeasurementType::AnalyzeWebRender.as_ref())?;
                metrics.measure_analysis_stats(sh, MeasurementType::AnalyzeDiesel.as_ref())?;
                metrics.measure_analysis_stats(sh, MeasurementType::AnalyzeHyper.as_ref())?;
                "all"
            }
        };

        let mut file =
            fs::File::options().write(true).create(true).open(format!("target/{name}.json"))?;
        writeln!(file, "{}", metrics.json())?;
        eprintln!("{metrics:#?}");
        Ok(())
    }
}

impl Metrics {
    fn measure_build(&mut self, sh: &Shell) -> anyhow::Result<()> {
        eprintln!("\nMeasuring build");
        cmd!(sh, "cargo fetch").run()?;

        let time = Instant::now();
        cmd!(sh, "cargo build --release --package rust-analyzer --bin rust-analyzer").run()?;
        let time = time.elapsed();
        self.report("build", time.as_millis() as u64, "ms".into());
        Ok(())
    }

    fn measure_rustc_tests(&mut self, sh: &Shell) -> anyhow::Result<()> {
        eprintln!("\nMeasuring rustc tests");

        cmd!(
            sh,
            "git clone --depth=1 --branch 1.76.0 https://github.com/rust-lang/rust.git --single-branch"
        )
        .run()?;

        let output = cmd!(sh, "./target/release/rust-analyzer rustc-tests ./rust").read()?;
        for (metric, value, unit) in parse_metrics(&output) {
            self.report(metric, value, unit.into());
        }
        Ok(())
    }

    fn measure_analysis_stats_self(&mut self, sh: &Shell) -> anyhow::Result<()> {
        self.measure_analysis_stats_path(sh, "self", ".")
    }
    fn measure_analysis_stats(&mut self, sh: &Shell, bench: &str) -> anyhow::Result<()> {
        self.measure_analysis_stats_path(
            sh,
            bench,
            &format!("./target/rustc-perf/collector/compile-benchmarks/{bench}"),
        )
    }
    fn measure_analysis_stats_path(
        &mut self,
        sh: &Shell,
        name: &str,
        path: &str,
    ) -> anyhow::Result<()> {
        assert!(Path::new(path).exists(), "unable to find bench in {path}");
        eprintln!("\nMeasuring analysis-stats/{name}");
        let output = cmd!(sh, "./target/release/rust-analyzer -q analysis-stats {path}").read()?;
        for (metric, value, unit) in parse_metrics(&output) {
            self.report(&format!("analysis-stats/{name}/{metric}"), value, unit.into());
        }
        Ok(())
    }
}

fn parse_metrics(output: &str) -> Vec<(&str, u64, &str)> {
    output
        .lines()
        .filter_map(|it| {
            let entry = it.split(':').collect::<Vec<_>>();
            match entry.as_slice() {
                ["METRIC", name, value, unit] => Some((*name, value.parse().unwrap(), *unit)),
                _ => None,
            }
        })
        .collect()
}

#[derive(Debug)]
struct Metrics {
    host: Host,
    timestamp: SystemTime,
    revision: String,
    perf_revision: String,
    metrics: BTreeMap<String, (u64, Unit)>,
}

#[derive(Debug)]
struct Host {
    os: String,
    cpu: String,
    mem: String,
}

impl Metrics {
    fn new(sh: &Shell) -> anyhow::Result<Metrics> {
        let host = Host::new(sh)?;
        let timestamp = SystemTime::now();
        let revision = cmd!(sh, "git rev-parse HEAD").read()?;
        let perf_revision = "a584462e145a0c04760fd9391daefb4f6bd13a99".into();
        Ok(Metrics { host, timestamp, revision, perf_revision, metrics: BTreeMap::new() })
    }

    fn report(&mut self, name: &str, value: u64, unit: Unit) {
        self.metrics.insert(name.into(), (value, unit));
    }

    fn json(&self) -> String {
        let mut buf = String::new();
        self.to_json(write_json::object(&mut buf));
        buf
    }

    fn to_json(&self, mut obj: write_json::Object<'_>) {
        self.host.to_json(obj.object("host"));
        let timestamp = self.timestamp.duration_since(UNIX_EPOCH).unwrap();
        obj.number("timestamp", timestamp.as_secs() as f64);
        obj.string("revision", &self.revision);
        obj.string("perf_revision", &self.perf_revision);
        let mut metrics = obj.object("metrics");
        for (k, (value, unit)) in &self.metrics {
            metrics.array(k).number(*value as f64).string(unit);
        }
    }
}

impl Host {
    fn new(sh: &Shell) -> anyhow::Result<Host> {
        if cfg!(not(target_os = "linux")) {
            return Ok(Host { os: "unknown".into(), cpu: "unknown".into(), mem: "unknown".into() });
        }

        let os = read_field(sh, "/etc/os-release", "PRETTY_NAME=")?.trim_matches('"').to_owned();

        let cpu = read_field(sh, "/proc/cpuinfo", "model name")?
            .trim_start_matches(':')
            .trim()
            .to_owned();

        let mem = read_field(sh, "/proc/meminfo", "MemTotal:")?;

        return Ok(Host { os, cpu, mem });

        fn read_field(sh: &Shell, path: &str, field: &str) -> anyhow::Result<String> {
            let text = sh.read_file(path)?;

            text.lines()
                .find_map(|it| it.strip_prefix(field))
                .map(|it| it.trim().to_owned())
                .ok_or_else(|| format_err!("can't parse {}", path))
        }
    }
    fn to_json(&self, mut obj: write_json::Object<'_>) {
        obj.string("os", &self.os).string("cpu", &self.cpu).string("mem", &self.mem);
    }
}
