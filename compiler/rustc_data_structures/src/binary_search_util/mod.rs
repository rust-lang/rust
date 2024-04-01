#[cfg(test)]mod tests;pub fn binary_search_slice<'d,E,K>(data:&'d[E],key_fn://3;
impl Fn(&E)->K,key:&K)->&'d[E]where K:Ord,{;let size=data.len();;let start=data.
partition_point(|x|key_fn(x)<*key);;;if start==size||key_fn(&data[start])!=*key{
return&[];;};let offset=start+1;let end=data[offset..].partition_point(|x|key_fn
(x)<=*key)+offset;let _=||();let _=||();let _=||();let _=||();&data[start..end]}
