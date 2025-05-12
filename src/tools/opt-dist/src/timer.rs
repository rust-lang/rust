use std::ops::{Deref, DerefMut};
use std::time::{Duration, SystemTime};

pub struct Timer {
    root: TimerSection,
}

impl Timer {
    pub fn new() -> Self {
        Timer { root: TimerSection::new(None) }
    }

    pub fn format_stats(&self) -> String {
        use std::fmt::Write;

        let mut items = Vec::new();
        for (name, child) in &self.root.children {
            match child {
                SectionEntry::SubSection(section) => {
                    section.collect_levels(0, name, &mut items);
                }
                SectionEntry::Duration(duration) => items.push((0, name, *duration)),
            }
        }

        let rows: Vec<(String, Duration)> = items
            .into_iter()
            .map(|(level, name, duration)| (format!("{}{name}:", "  ".repeat(level)), duration))
            .collect();

        let total_duration = self.total_duration();
        let total_duration_label = "Total duration:".to_string();

        const SPACE_AFTER_LABEL: usize = 2;
        let max_label_length = 16.max(rows.iter().map(|(label, _)| label.len()).max().unwrap_or(0))
            + SPACE_AFTER_LABEL;

        let table_width = max_label_length + 23;
        let divider = "-".repeat(table_width);

        let mut output = String::new();
        writeln!(output, "{divider}").unwrap();
        for (label, duration) in rows {
            let pct = (duration.as_millis() as f64 / total_duration.as_millis() as f64) * 100.0;
            let duration_fmt = format!("{:>12.2}s ({pct:>5.2}%)", duration.as_secs_f64());
            writeln!(output, "{label:<0$} {duration_fmt}", max_label_length).unwrap();
        }
        output.push('\n');

        let total_duration = Duration::new(total_duration.as_secs(), 0);
        let total_duration = format!(
            "{:>1$}",
            humantime::format_duration(total_duration).to_string(),
            table_width - total_duration_label.len()
        );
        writeln!(output, "{total_duration_label}{total_duration}").unwrap();

        writeln!(output, "{divider}").unwrap();
        output
    }
}

impl Deref for Timer {
    type Target = TimerSection;

    fn deref(&self) -> &Self::Target {
        &self.root
    }
}

impl DerefMut for Timer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.root
    }
}

pub struct TimerSection {
    name: Option<String>,
    children: Vec<(String, SectionEntry)>,
    duration_excluding_children: Duration,
}

impl TimerSection {
    pub fn new(name: Option<String>) -> Self {
        TimerSection {
            name,
            children: Default::default(),
            duration_excluding_children: Duration::ZERO,
        }
    }

    pub fn section<F: FnOnce(&mut TimerSection) -> anyhow::Result<R>, R>(
        &mut self,
        name: &str,
        func: F,
    ) -> anyhow::Result<R> {
        let full_name = match &self.name {
            Some(current_name) => {
                format!("{current_name} > {name}")
            }
            None => name.to_string(),
        };
        log::info!("Section `{full_name}` starts");
        let mut child = TimerSection {
            name: Some(full_name.clone()),
            children: Default::default(),
            duration_excluding_children: Duration::ZERO,
        };

        let start = SystemTime::now();
        let result = func(&mut child);
        let duration = start.elapsed().unwrap();

        let msg = match result {
            Ok(_) => "OK",
            Err(_) => "FAIL",
        };

        child.duration_excluding_children = duration.saturating_sub(child.total_duration());

        log::info!("Section `{full_name}` ended: {msg} ({:.2}s)`", duration.as_secs_f64());
        self.children.push((name.to_string(), SectionEntry::SubSection(child)));
        result
    }

    pub fn add_duration(&mut self, name: &str, duration: Duration) {
        self.children.push((name.to_string(), SectionEntry::Duration(duration)));
    }

    fn total_duration(&self) -> Duration {
        self.duration_excluding_children
            + self.children.iter().map(|(_, child)| child.total_duration()).sum::<Duration>()
    }

    fn collect_levels<'a>(
        &'a self,
        level: usize,
        name: &'a str,
        items: &mut Vec<(usize, &'a str, Duration)>,
    ) {
        items.push((level, name, self.total_duration()));
        for (name, child) in &self.children {
            match &child {
                SectionEntry::Duration(duration) => {
                    items.push((level + 1, name, *duration));
                }
                SectionEntry::SubSection(section) => {
                    section.collect_levels(level + 1, name, items);
                }
            }
        }
    }
}

enum SectionEntry {
    Duration(Duration),
    SubSection(TimerSection),
}

impl SectionEntry {
    fn total_duration(&self) -> Duration {
        match self {
            SectionEntry::Duration(duration) => *duration,
            SectionEntry::SubSection(timer) => timer.total_duration(),
        }
    }
}
