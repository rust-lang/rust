//@ edition:2021

type TranslateFn = Box<dyn Fn(String, String) -> String>;

pub struct DeviceCluster {
    devices: Vec<Device>,
}

impl DeviceCluster {
    pub async fn do_something(&mut self) -> Result<String, Box<dyn std::error::Error>> {
        let mut last_error: Box<dyn std::error::Error>;

        for device in &mut self.devices {
            match device.do_something().await {
                Ok(info) => {
                    return Ok(info);
                }
                Err(e) => {}
            }
        }

        Err(last_error)
        //~^ ERROR used binding `last_error` isn't initialized
    }
}

pub struct Device {
    translate_fn: Option<TranslateFn>,
}

impl Device {
    pub async fn do_something(&mut self) -> Result<String, Box<dyn std::error::Error>> {
        Ok(String::from(""))
    }
}

fn main() {}
