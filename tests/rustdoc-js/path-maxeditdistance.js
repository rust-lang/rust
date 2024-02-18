// exact-check

const EXPECTED = [
    {
        'query': 'xxxxxxxxxxx::hocuspocusprestidigitation',
        // do not match abracadabra::hocuspocusprestidigitation
        'others': [],
    },
    {
        // exact match
        'query': 'abracadabra::hocuspocusprestidigitation',
        'others': [
            { 'path': 'abracadabra', 'name': 'HocusPocusPrestidigitation' },
        ],
    },
    {
        // swap br/rb; that's edit distance 2, where maxPathEditDistance = 3 (11 / 3)
        'query': 'arbacadarba::hocuspocusprestidigitation',
        'others': [
            { 'path': 'abracadabra', 'name': 'HocusPocusPrestidigitation' },
        ],
    },
    {
        // truncate 5 chars, where maxEditDistance = 7 (21 / 3)
        'query': 'abracadarba::hocusprestidigitation',
        'others': [
            { 'path': 'abracadabra', 'name': 'HocusPocusPrestidigitation' },
        ],
    },
    {
        // truncate 9 chars, where maxEditDistance = 5 (17 / 3)
        'query': 'abracadarba::hprestidigitation',
        'others': [],
    },
];
