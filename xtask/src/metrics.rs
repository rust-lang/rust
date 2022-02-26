use std::{
    collections::BTreeMap,
    env,
    io::Write as _,
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{bail, format_err, Result};
use xshell::{cmd, mkdir_p, pushd, pushenv, read_file, rm_rf};

use crate::flags;

type Unit = String;

impl flags::Metrics {
    pub(crate) fn run(self) -> Result<()> {
        let mut metrics = Metrics::new()?;
        if !self.dry_run {
            rm_rf("./target/release")?;
        }
        if !Path::new("./target/rustc-perf").exists() {
            mkdir_p("./target/rustc-perf")?;
            cmd!("git clone https://github.com/rust-lang/rustc-perf.git ./target/rustc-perf")
                .run()?;
        }
        {
            let _d = pushd("./target/rustc-perf")?;
            let revision = &metrics.perf_revision;
            cmd!("git reset --hard {revision}").run()?;
        }

        let _env = pushenv("RA_METRICS", "1");

        {
            // https://github.com/rust-analyzer/rust-analyzer/issues/9997
            let _d = pushd("target/rustc-perf/collector/benchmarks/webrender")?;
            cmd!("cargo update -p url --precise 1.6.1").run()?;
        }
        metrics.measure_build()?;
        metrics.measure_analysis_stats_self()?;
        metrics.measure_analysis_stats("ripgrep")?;
        metrics.measure_analysis_stats("webrender")?;
        metrics.measure_analysis_stats("diesel/diesel")?;

        if !self.dry_run {
            let _d = pushd("target")?;
            let metrics_token = env::var("METRICS_TOKEN").unwrap();
            cmd!(
                "git clone --depth 1 https://{metrics_token}@github.com/rust-analyzer/metrics.git"
            )
            .run()?;
            let _d = pushd("metrics")?;

            let mut file = std::fs::OpenOptions::new().append(true).open("metrics.json")?;
            writeln!(file, "{}", metrics.json())?;
            cmd!("git add .").run()?;
            cmd!("git -c user.name=Bot -c user.email=dummy@example.com commit --message 📈")
                .run()?;
            cmd!("git push origin master").run()?;
        }
        eprintln!("{:#?}", metrics);
        Ok(())
    }
}

impl Metrics {
    fn measure_build(&mut self) -> Result<()> {
        eprintln!("\nMeasuring build");
        cmd!("cargo fetch").run()?;

        let time = Instant::now();
        cmd!("cargo build --release --package rust-analyzer --bin rust-analyzer").run()?;
        let time = time.elapsed();
        self.report("build", time.as_millis() as u64, "ms".into());
        Ok(())
    }
    fn measure_analysis_stats_self(&mut self) -> Result<()> {
        self.measure_analysis_stats_path("self", ".")
    }
    fn measure_analysis_stats(&mut self, bench: &str) -> Result<()> {
        self.measure_analysis_stats_path(
            bench,
            &format!("./target/rustc-perf/collector/benchmarks/{}", bench),
        )
    }
    fn measure_analysis_stats_path(&mut self, name: &str, path: &str) -> Result<()> {
        eprintln!("\nMeasuring analysis-stats/{}", name);
        let output = cmd!("./target/release/rust-analyzer -q analysis-stats --memory-usage {path}")
            .read()?;
        for (metric, value, unit) in parse_metrics(&output) {
            self.report(&format!("analysis-stats/{}/{}", name, metric), value, unit.into());
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
    fn new() -> Result<Metrics> {
        let host = Host::new()?;
        let timestamp = SystemTime::now();
        let revision = cmd!("git rev-parse HEAD").read()?;
        let perf_revision = "c52ee623e231e7690a93be88d943016968c1036b".into();
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
    fn new() -> Result<Host> {
        if cfg!(not(target_os = "linux")) {
            bail!("can only collect metrics on Linux ");
        }

        let os = read_field("/etc/os-release", "PRETTY_NAME=")?.trim_matches('"').to_string();

        let cpu =
            read_field("/proc/cpuinfo", "model name")?.trim_start_matches(':').trim().to_string();

        let mem = read_field("/proc/meminfo", "MemTotal:")?;

        return Ok(Host { os, cpu, mem });

        fn read_field(path: &str, field: &str) -> Result<String> {
            let text = read_file(path)?;

            let line = text
                .lines()
                .find(|it| it.starts_with(field))
                .ok_or_else(|| format_err!("can't parse {}", path))?;
            Ok(line[field.len()..].trim().to_string())
        }
    }
    fn to_json(&self, mut obj: write_json::Object<'_>) {
        obj.string("os", &self.os).string("cpu", &self.cpu).string("mem", &self.mem);
    }
}
